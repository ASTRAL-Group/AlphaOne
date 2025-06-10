set -ex

export CUDA_VISIBLE_DEVICES="5,6"
MODEL_NAME_OR_PATH="Qwen/QwQ-32B"
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/s1/

# Template for DeepSeek evaluation.
# For detailed settings and customization, refer to the 'utils.py' file.
PROMPT_TYPE="qwen25-math-cot"
# Choose the dataset to use for evaluation. Options include aime24, amc23, minerva_math, math500, olympiadbench
DATA_NAME="aime24"

SPLIT="test"
NUM_TEST_SAMPLE=-1
echo "Launch s1"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_efficient.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 8192\
    --use_s1 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

DATA_NAME="amc23"

SPLIT="test"
NUM_TEST_SAMPLE=-1
echo "Launch s1"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_efficient.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 8192\
    --use_s1 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

DATA_NAME="minerva_math"

SPLIT="test"
NUM_TEST_SAMPLE=-1
echo "Launch s1"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_efficient.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 8192\
    --use_s1 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

DATA_NAME="math500"

SPLIT="test"
NUM_TEST_SAMPLE=-1
echo "Launch s1"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_efficient.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 8192\
    --use_s1 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

DATA_NAME="olympiadbench"

SPLIT="test"
NUM_TEST_SAMPLE=-1
echo "Launch s1"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_efficient.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 8192\
    --use_s1 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite



