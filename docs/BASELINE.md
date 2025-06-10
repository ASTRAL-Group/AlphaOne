## Baselines Evaluation

We provide evaluation scripts for evaluating baselines on math and science benchmarks, such as **AIME24**, **AMC23**, **Minerva-Math**, **MATH500** and **OlympiadBench**. For **LiveCodeBench**, a dedicated evaluation script is provided.


### Base Model

- For **DeepSeek-R1-Distill-Qwen-1.5B** on Math & Science Benchmarks:
```bash
./scripts/base/eval_base_deepseek_1_5b.sh
```
- For **DeepSeek-R1-Distill-Qwen-7B** on Math & Science Benchmarks:
```bash
./scripts/base/eval_base_deepseek_7b.sh
```
- For **Qwen QwQ-32B** on Math & Science Benchmarks:
```bash
./scripts/base/eval_base_qwq.sh
```
- For **LiveCodeBench**:
```bash
./scripts/base/eval_base_code.sh
```
### s1
- For **DeepSeek-R1-Distill-Qwen-1.5B** on Math & Science Benchmarks:
```bash
./scripts/s1/eval_s1_deepseek_1_5b.sh
```
- For **DeepSeek-R1-Distill-Qwen-7B** on Math & Science Benchmarks:
```bash
./scripts/s1/eval_s1_deepseek_7b.sh
```
- For **Qwen QwQ-32B** on Math & Science Benchmarks:
```bash
./scripts/s1/eval_s1_qwq.sh
```
- For **LiveCodeBench**:
```bash
./scripts/s1/eval_s1_code.sh
```

### Chain of Draft (CoD)
- For **DeepSeek-R1-Distill-Qwen-1.5B** on Math & Science Benchmarks:
```bash
./scripts/cod/eval_cod_deepseek_1_5b.sh
```
- For **DeepSeek-R1-Distill-Qwen-7B** on Math & Science Benchmarks:
```bash
./scripts/cod/eval_cod_deepseek_7b.sh
```
- For **Qwen QwQ-32B** on Math & Science Benchmarks:
```bash
./scripts/cod/eval_cod_qwq.sh
```
- For **LiveCodeBench**:
```bash
./scripts/cod/eval_cod_code.sh
```

