import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from functools import partial
import multiprocessing as mp


import sys
import os
import gc
from code_evaluation import codegen_metrics, load_code_generation_dataset, get_deepseekcode_question_template_answer, get_deepseekcode_question_template_answer_cod, extract_code, extract_instance_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
    if max_len <= 0 or len(token_ids) <= max_len:
        logits[adjust_ids.to(logits.device)] += values
    return logits



def main(args):
    random.seed(42)

    print("Loading data...")

    if args.release == "v5-v1":
        benchmark_v5 = load_code_generation_dataset(release_version="release_v5")
        benchmark_v1 = load_code_generation_dataset(release_version="release_v1")
        benchmark = [d for d in benchmark_v5 if d not in benchmark_v1]
        assert len(benchmark)==480
    else:
        benchmark = load_code_generation_dataset(release_version=args.release)
    
    if args.max_examples and len(benchmark) > args.max_examples:
        benchmark = benchmark[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for i, example in enumerate(benchmark):
        if args.use_cod:
            prompt =  get_deepseekcode_question_template_answer_cod(example)
        else:
            prompt =  get_deepseekcode_question_template_answer(example)
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    model = LLM(model=args.model_name_or_path, 
                quantization="fp8",
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path, swap_space=16, gpu_memory_utilization=0.98, enable_lora=args.peft is not None, tensor_parallel_size=torch.cuda.device_count(), max_lora_rank=128, max_model_len=args.max_tokens+2000)

    if not args.logit_adjustment:

        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens)
    else:
        vocab = tokenizer.get_vocab()
        logit_adjustment_tokens = torch.LongTensor([vocab[token] for token in vocab.keys() if any([x in token for x in args.logit_adjustment_tokens])]).to("cuda")
        logit_adjustment_process = partial(logit_adjustment, adjust_ids=logit_adjustment_tokens, values=args.logit_adjustment_value, max_len=args.logit_adjustment_max_len)
        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens,
                                        logits_processors=[logit_adjustment_process]
                                        )
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|end▁of▁sentence|>"]
    if args.peft is not None:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=LoRARequest("lora_path", 1, lora_path=args.peft))
    else:
        if args.use_s1:
            
            gen_output = model.generate(prompts, SamplingParams(
                                    temperature=0,
                                    #   top_p=args.top_p,
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    n=1, stop=stop_words+["</think>"],
                                    include_stop_str_in_output=True,))
            # First time 
            outputs = []
            remaining_tokens = []
            for index in range(len(prompts)):
                response = gen_output[index].outputs[0].text
                response = response.replace("</think>", "Wait")
                outputs.append(response)
                prompts[index] += response
                remaining_tokens.append(max(1, args.max_tokens-len(gen_output[index].outputs[0].token_ids)))
            
            sampling_params_list = [SamplingParams(
                                temperature=0,
                                #   top_p=args.top_p,
                                max_tokens=remaining_token,
                                seed=args.seed,
                                n=1, stop=stop_words+["</think>"],
                                include_stop_str_in_output=True,) for remaining_token in remaining_tokens]
            gen_output = model.generate(prompts, sampling_params_list)

            # Second time 
            for index in range(len(prompts)):
                response = gen_output[index].outputs[0].text
                response = response.replace("</think>", "Wait")
                # print("Index: ", index, " | Response: ", response)
                outputs[index] += response
                prompts[index] += response
                remaining_tokens[index] = max(1, remaining_tokens[index]-len(gen_output[index].outputs[0].token_ids))
            
            sampling_params_list = [SamplingParams(
                                temperature=0,
                                #   top_p=args.top_p,
                                max_tokens=remaining_token,
                                seed=args.seed,
                                n=1, stop=stop_words,
                                include_stop_str_in_output=True,) for remaining_token in remaining_tokens]
            # Third time,
            gen_output = model.generate(prompts, sampling_params_list)
            for index in range(len(prompts)):
                response = gen_output[index].outputs[0].text
                outputs[index] += response
        elif args.use_wait_more:
            from itertools import chain
            import numpy as np

            class NewlineWait:
                def __init__(self, tokenizer, max_token_per_call=0, threshold=0):
                    self.newline_ids = tokenizer(["\n\n", ",\n\n", ".\n\n", "]\n\n",
                                                ")\n\n", "],\n\n", "].\n\n", "].\n\n",
                                                ").\n\n", ".)\n\n"], add_special_tokens=False).input_ids
                    self.newline_ids = list(chain.from_iterable(self.newline_ids))
                    self.wait_id = tokenizer.encode("Wait", add_special_tokens=False)[0]
                    self.think_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
                    self.max_token_per_call = max_token_per_call
                    self.threshold = threshold

                def __call__(self, token_ids, logits):

                    if len(token_ids)<2:
                        return logits
                    
                    remaining_tokens = self.max_token_per_call - len(token_ids)
                    if remaining_tokens >= self.threshold and token_ids[-1] in self.newline_ids:
                        p_wait = (remaining_tokens - self.threshold) / (self.max_token_per_call - self.threshold)
                        if random.random() < p_wait:
                            logits.fill_(-float("inf"))
                            logits[self.wait_id] = 0.0
                    return logits
                
            if "1.5B" in args.model_name_or_path:
                args.threshold = args.max_tokens - args.alpha * 2500
            elif "7B" in args.model_name_or_path:
                args.threshold = args.max_tokens - args.alpha * 3150
            elif "32B" in args.model_name_or_path:
                args.threshold = args.max_tokens - args.alpha * 4800
            logits_processor = NewlineWait(model.get_tokenizer(), max_token_per_call=args.max_tokens, threshold=args.threshold)
            gen_output = model.generate(prompts, SamplingParams(
                                temperature=0,
                                #   top_p=args.top_p,
                                max_tokens=args.max_tokens - args.threshold, 
                                seed=args.seed,
                                n=1, stop=stop_words,
                                include_stop_str_in_output=True,
                                logits_processors=[logits_processor]))

            outputs = []
            remaining_tokens_ = []
            prompts_ = []
            for index in range(len(prompts)):
                outputs.append(gen_output[index].outputs[0].text)
                prompts_.append(prompts[index]+gen_output[index].outputs[0].text)
                remaining_tokens_.append(args.max_tokens - len(gen_output[index].outputs[0].token_ids))
            
            assert len(prompts_) == len(remaining_tokens_)
            stop_at_wait_words = ["Wait", ".Wait", "Wait, ", "Wait,",
                                # "wait",  ".wait", , "wait, ", "wait,"
                                ]
           
            active_traj = np.ones(len(prompts_))
            while 1:
                sampling_params_list = [SamplingParams(
                                    temperature=0,
                                    #   top_p=args.top_p,
                                    max_tokens=remaining_token_, seed=args.seed,
                                    n=1, stop=stop_words+stop_at_wait_words,
                                    include_stop_str_in_output=True,) for remaining_token_ in remaining_tokens_]
                
                input_prompts_while = [q for f, q in zip(active_traj, prompts_) if f]
                input_sampling_params_list = [q for f, q in zip(active_traj, sampling_params_list) if f]
                gen_output = model.generate(input_prompts_while, input_sampling_params_list)
                
                i = 0
                for index in range(len(prompts_)):
                    if active_traj[index] == 1:
                        response = gen_output[i].outputs[0].text
                        response = response.replace("\nWait", "</think>")
                        response = response.replace("Wait", "</think>")
                        
                        # print("Index: ", index, " | Response: ", response)
                        outputs[index] += response
                        prompts_[index] += response
                        remaining_tokens_[index] = max(1, remaining_tokens_[index]-len(gen_output[i].outputs[0].token_ids))

                        if response.endswith(tuple(stop_words)) or remaining_tokens_[index]==1 or response=="":
                            active_traj[index] = 0
                        i += 1
                    
                print("ACTIVTE TRAJ: ", sum(active_traj))
                if sum(active_traj)==0:
                    break
        else:
            outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

    results = []
    if not (args.use_s1 or args.use_wait_more):
        for index, output in enumerate(outputs):
            attempts = []
            for ith_output in output.outputs:
                attempts.append(ith_output.text)
            results.append(attempts)
    else:
        for output in outputs:
            results.append([output])
        # results = outputs

    output_token_lengths = []
    for idx_prompt, full_output in enumerate(results):
        full_output = full_output[0]
        current_token_length = len(tokenizer.encode(full_output)) if tokenizer else len(full_output.split())
        output_token_lengths.append(current_token_length)
        average_token_length = sum(output_token_lengths) / len(output_token_lengths)

    print(f"Average Token Length: {average_token_length:.2f}")

    combined_results = [
        (
            outputs_list,
            [extract_code(output) for output in outputs_list],
        )
        for outputs_list in results
    ]

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            benchmark, combined_results
        )
    ]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as f:
        json.dump(save_results, f, indent=4)


    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=12,
        timeout=10,
    )

    print(metrics[0]["pass@1"])

    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]
    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metadatas
        )
    ]

    with open(os.path.join(args.save_dir, "metrics.jsonl"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(args.save_dir, "code_eval.jsonl"), "w") as f:
        json.dump(save_eval_results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--release",
        type=str,
        default="release_v1",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logit_adjustment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logit_adjustment_tokens",
        type=str,
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--logit_adjustment_value",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--logit_adjustment_max_len",
        type=int,
        default=-1
    )
    parser.add_argument("--use_wait_more", action="store_true")
    parser.add_argument("--use_s1", action="store_true")
    parser.add_argument("--use_cod", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=42)

    
    args = parser.parse_args()

    if args.logit_adjustment:
        name = "_".join(args.logit_adjustment_tokens)+f"_value_{args.logit_adjustment_value}"
        if args.logit_adjustment_max_len>0:
            name += f"_first{args.logit_adjustment_max_len}"
        
        args.save_dir = os.path.join(args.save_dir, "logit-adjustment", name)
        
    mp.set_start_method("spawn", force=True)
    main(args)

        
