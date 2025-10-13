__version__ = "0.1.4"


def __getattr__(name: str):  # type: ignore
    if name == "LLMSQLVLLMInference":
        from .inference.inference import LLMSQLVLLMInference

        return LLMSQLVLLMInference
    elif name == "LLMSQLEvaluator":
        from .evaluation.evaluator import LLMSQLEvaluator

        return LLMSQLEvaluator
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


__all__ = ["LLMSQLEvaluator", "LLMSQLVLLMInference"]
