"""
LLMSQL — A Text2SQL benchmark for evaluation of Large Language Models
"""

__version__ = "0.1.13"

import platform

VLLM_SUPPORTED_PLATFORMS = ["Linux"]
CURRENT_OS = platform.system()


def __getattr__(name: str):  # type: ignore
    if name == "evaluate":
        from .evaluation.evaluate import evaluate
        return evaluate
    
    elif name == "inference_vllm":
        if CURRENT_OS not in VLLM_SUPPORTED_PLATFORMS:
            raise RuntimeError(
                f"vLLM backend is not supported on {CURRENT_OS}. "
                f"Supported platforms: {', '.join(VLLM_SUPPORTED_PLATFORMS)}. "
                "Please use --method transformers."
            )

        try:
            from .inference.inference_vllm import inference_vllm
            return inference_vllm
        
        except ModuleNotFoundError as e:
            if "vllm" in str(e):
                raise ImportError(
                    "The vLLM backend is not installed. "
                    "Install it with: pip install llmsql[vllm]"
                ) from e
            raise

    elif name == "inference_transformers":
        from .inference.inference_transformers import inference_transformers

        return inference_transformers
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


__all__ = ["evaluate", "inference_vllm", "inference_transformers"]
