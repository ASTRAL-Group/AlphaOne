VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 python code_eval_efficient.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_dir outputs/code/DeepSeek-R1-Distill-Qwen-1.5B/base \
    --max_tokens 8192 \
    --use_chat_format \
    --remove_bos \

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 python code_eval_efficient.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --save_dir outputs/code/DeepSeek-R1-Distill-Qwen-7B/base \
    --max_tokens 8192 \
    --use_chat_format \
    --remove_bos \

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1 python code_eval_efficient.py \
    --model_name_or_path Qwen/QwQ-32B \
    --save_dir outputs/code/QwQ-32B/base \
    --max_tokens 8192 \
    --use_chat_format \
    --remove_bos \



