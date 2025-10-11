"""Tests for llmsql package initialization and lazy imports."""

import pytest


class TestLazyImport:
    """Test lazy import mechanism in __init__.py."""

    def test_lazy_import_evaluator(self) -> None:
        """Test that LLMSQLEvaluator can be imported via lazy loading."""
        from llmsql import LLMSQLEvaluator

        assert LLMSQLEvaluator is not None
        # Verify it's the correct class
        assert LLMSQLEvaluator.__name__ == "LLMSQLEvaluator"

    def test_lazy_import_inference_skipped_if_vllm_missing(self) -> None:
        """Test that LLMSQLVLLMInference import fails gracefully without vllm."""
        try:
            from llmsql import LLMSQLVLLMInference

            # If vllm is installed, should succeed
            assert LLMSQLVLLMInference is not None
        except ModuleNotFoundError as e:
            # If vllm not installed, should raise ModuleNotFoundError
            assert "vllm" in str(e).lower()

    def test_invalid_attribute_raises_error(self) -> None:
        """Test that accessing invalid attribute raises AttributeError."""
        import llmsql

        with pytest.raises(
            AttributeError, match="module .* has no attribute 'NonExistentClass'"
        ):
            _ = llmsql.NonExistentClass  # type: ignore

    def test_version_attribute(self) -> None:
        """Test that __version__ is accessible."""
        import llmsql

        assert hasattr(llmsql, "__version__")
        assert isinstance(llmsql.__version__, str)
        # Should match semantic versioning pattern
        assert len(llmsql.__version__.split(".")) >= 2

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        import llmsql

        assert hasattr(llmsql, "__all__")
        assert "LLMSQLEvaluator" in llmsql.__all__
        assert "LLMSQLVLLMInference" in llmsql.__all__
