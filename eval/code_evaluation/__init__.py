# 在 __init__.py 里
from .code_utils import CodeGenerationProblem, load_code_generation_dataset, get_deepseekcode_question_template_answer,get_deepseekcode_question_template_answer_cod, extract_code, extract_instance_results
from .compute_code_generation_metrics import codegen_metrics

# 明确指定 __all__，控制暴露的 API
__all__ = [
    "CodeGenerationProblem",
    "load_code_generation_dataset",
    "get_deepseekcode_question_template_answer",
    "get_deepseekcode_question_template_answer_cod",
    "extract_code",
    "extract_instance_results",
    "codegen_metrics"
]
