import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor


def evaluate_avg(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": np.mean(mean_score)
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    return samples, result_json

def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    
    # First evaluate all individual predictions to get scores
    params = [(idx, pred, sample['gt']) 
             for idx, sample in enumerate(samples) 
             for pred in sample['pred']]

    individual_scores = []
    timeout_cnt = 0

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(params), desc="Evaluate Individual Predictions") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    individual_scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    individual_scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1)

    # Assign individual scores back to samples
    score_idx = 0
    for sample in samples:
        num_preds = len(sample['pred'])
        sample['score'] = individual_scores[score_idx:score_idx+num_preds]
        score_idx += num_preds

    # Calculate majority vote results (final_score)
    empty_samples = 0
    for sample in samples:
        if not sample['pred']:
            sample['final_score'] = False
            empty_samples += 1
            continue
        
        # Get scores for this sample's predictions
        pred_scores = sample['score']
        
        # Group predictions by their string representation and sum their scores
        from collections import defaultdict
        pred_groups = defaultdict(list)
        for pred, score in zip(sample['pred'], pred_scores):
            pred_groups[pred].append(score)
        
        # Find the group with most votes (majority)
        majority_pred = max(pred_groups.items(), key=lambda x: len(x[1]))[0]
        
        # Calculate the correctness of majority vote
        # If any of the majority predictions is correct, count as correct
        majority_correct = any(pred_groups[majority_pred])
        sample['final_score'] = majority_correct

    # Calculate statistics
    final_scores = [s['final_score'] for s in samples if 'final_score' in s]
    acc = np.round(np.mean(final_scores) * 100, decimals=1) if final_scores else 0.0

    result_json = {
        "num_samples": len(samples),
        "num_individual_scores": len(individual_scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": empty_samples,
        "acc": acc,  # This is now based on final_score (majority vote)
        "individual_acc": np.round(np.mean(individual_scores) * 100, decimals=1) if individual_scores else 0.0
    }

    # each type score (based on final_score)
    if "type" in samples[0]:
        type_scores = {}
        type_individual_scores = {}
        
        for sample in samples:
            # Track final scores by type
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['final_score'])
            
            # Track individual scores by type
            if sample['type'] not in type_individual_scores:
                type_individual_scores[sample['type']] = []
            type_individual_scores[sample['type']].extend(sample['score'])
        
        # Calculate type accuracies
        result_json['type_acc'] = {
            k: np.round(np.array(v).mean() * 100, decimals=1) 
            for k, v in sorted(type_scores.items(), key=lambda item: item[0])
        }
        result_json['type_individual_acc'] = {
            k: np.round(np.array(v).mean() * 100, decimals=1) 
            for k, v in sorted(type_individual_scores.items(), key=lambda item: item[0])
        }

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
