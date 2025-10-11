"""Tests for llmsql.utils.utils module."""

import json
from pathlib import Path

import pytest

from llmsql.prompts.prompts import (
    build_prompt_0shot,
    build_prompt_1shot,
    build_prompt_5shot,
)
from llmsql.utils.utils import (
    choose_prompt_builder,
    load_jsonl,
    overwrite_jsonl,
    save_jsonl_lines,
)


class TestLoadJSONL:
    """Test cases for load_jsonl function."""

    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        """Test loading a valid JSONL file."""
        file_path = tmp_path / "test.jsonl"
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        # Write test data
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        # Load and verify
        result = load_jsonl(str(file_path))
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[0]["name"] == "Alice"
        assert result[1]["id"] == 2
        assert result[1]["name"] == "Bob"

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Test loading an empty file."""
        file_path = tmp_path / "empty.jsonl"
        file_path.write_text("")

        result = load_jsonl(str(file_path))
        assert result == []

    def test_skip_blank_lines(self, tmp_path: Path) -> None:
        """Test that blank and whitespace lines are skipped."""
        file_path = tmp_path / "blanks.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('{"id": 1}\n')
            f.write('\n')  # blank line
            f.write('   \n')  # whitespace line
            f.write('{"id": 2}\n')
            f.write('\t\n')  # tab line

        result = load_jsonl(str(file_path))
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_invalid_json_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid JSON raises an exception."""
        file_path = tmp_path / "invalid.jsonl"
        file_path.write_text('{"valid": true}\n{invalid json}\n')

        with pytest.raises(json.JSONDecodeError):
            load_jsonl(str(file_path))

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Test loading JSONL with Unicode/emoji content."""
        file_path = tmp_path / "unicode.jsonl"
        data = [{"text": "Hello 世界 🌍"}, {"emoji": "🎉🎊"}]

        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        result = load_jsonl(str(file_path))
        assert len(result) == 2
        assert result[0]["text"] == "Hello 世界 🌍"
        assert result[1]["emoji"] == "🎉🎊"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_jsonl(str(tmp_path / "nonexistent.jsonl"))

    def test_nested_json_structures(self, tmp_path: Path) -> None:
        """Test loading JSONL with nested JSON objects."""
        file_path = tmp_path / "nested.jsonl"
        data = [
            {"id": 1, "details": {"name": "Alice", "age": 30}},
            {"id": 2, "items": [1, 2, 3]},
        ]

        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        result = load_jsonl(str(file_path))
        assert result[0]["details"]["name"] == "Alice"
        assert result[1]["items"] == [1, 2, 3]


class TestSaveJSONLLines:
    """Test cases for save_jsonl_lines function."""

    def test_write_single_item(self, tmp_path: Path) -> None:
        """Test writing a single item to JSONL file."""
        file_path = tmp_path / "output.jsonl"
        items = [{"id": 1, "name": "Alice"}]

        save_jsonl_lines(str(file_path), items)

        # Verify file content
        result = load_jsonl(str(file_path))
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_write_multiple_items(self, tmp_path: Path) -> None:
        """Test writing multiple items."""
        file_path = tmp_path / "output.jsonl"
        items = [{"id": 1}, {"id": 2}, {"id": 3}]

        save_jsonl_lines(str(file_path), items)

        result = load_jsonl(str(file_path))
        assert len(result) == 3

    def test_append_behavior(self, tmp_path: Path) -> None:
        """Test that save_jsonl_lines appends to existing file."""
        file_path = tmp_path / "output.jsonl"

        # Write first batch
        save_jsonl_lines(str(file_path), [{"id": 1}])
        # Append second batch
        save_jsonl_lines(str(file_path), [{"id": 2}])

        result = load_jsonl(str(file_path))
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_unicode_ensure_ascii_false(self, tmp_path: Path) -> None:
        """Test that Unicode characters are preserved (ensure_ascii=False)."""
        file_path = tmp_path / "unicode.jsonl"
        items = [{"text": "日本語 🇯🇵"}]

        save_jsonl_lines(str(file_path), items)

        # Read raw content
        content = file_path.read_text(encoding="utf-8")
        # Should contain actual Unicode, not escaped
        assert "日本語" in content
        assert "🇯🇵" in content

    def test_empty_iterable(self, tmp_path: Path) -> None:
        """Test saving empty iterable creates empty file or does nothing."""
        file_path = tmp_path / "empty.jsonl"
        save_jsonl_lines(str(file_path), [])

        # File should exist but be empty (or not exist)
        if file_path.exists():
            assert file_path.read_text() == ""

    def test_generator_input(self, tmp_path: Path) -> None:
        """Test that function works with generator (Iterable)."""
        file_path = tmp_path / "generator.jsonl"

        def item_generator():
            for i in range(3):
                yield {"id": i}

        save_jsonl_lines(str(file_path), item_generator())

        result = load_jsonl(str(file_path))
        assert len(result) == 3


class TestOverwriteJSONL:
    """Test cases for overwrite_jsonl function."""

    def test_create_new_file(self, tmp_path: Path) -> None:
        """Test creating a new empty file."""
        file_path = tmp_path / "new.jsonl"
        overwrite_jsonl(str(file_path))

        assert file_path.exists()
        assert file_path.read_text() == ""

    def test_overwrite_existing_file(self, tmp_path: Path) -> None:
        """Test overwriting existing file makes it empty."""
        file_path = tmp_path / "existing.jsonl"
        file_path.write_text('{"old": "data"}\n')

        overwrite_jsonl(str(file_path))

        assert file_path.exists()
        assert file_path.read_text() == ""

    def test_create_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created if missing."""
        file_path = tmp_path / "subdir" / "nested" / "file.jsonl"

        overwrite_jsonl(str(file_path))

        assert file_path.exists()
        assert file_path.parent.exists()

    def test_idempotent_operation(self, tmp_path: Path) -> None:
        """Test that calling overwrite_jsonl multiple times is safe."""
        file_path = tmp_path / "test.jsonl"

        overwrite_jsonl(str(file_path))
        overwrite_jsonl(str(file_path))
        overwrite_jsonl(str(file_path))

        assert file_path.exists()
        assert file_path.read_text() == ""


class TestChoosePromptBuilder:
    """Test cases for choose_prompt_builder function."""

    def test_shots_0_returns_0shot_builder(self) -> None:
        """Test that shots=0 returns build_prompt_0shot."""
        builder = choose_prompt_builder(0)
        assert builder is build_prompt_0shot

    def test_shots_1_returns_1shot_builder(self) -> None:
        """Test that shots=1 returns build_prompt_1shot."""
        builder = choose_prompt_builder(1)
        assert builder is build_prompt_1shot

    def test_shots_5_returns_5shot_builder(self) -> None:
        """Test that shots=5 returns build_prompt_5shot."""
        builder = choose_prompt_builder(5)
        assert builder is build_prompt_5shot

    def test_invalid_shots_2_raises_error(self) -> None:
        """Test that shots=2 raises ValueError."""
        with pytest.raises(ValueError, match="shots must be one of"):
            choose_prompt_builder(2)

    def test_invalid_shots_negative_raises_error(self) -> None:
        """Test that negative shots raises ValueError."""
        with pytest.raises(ValueError, match="shots must be one of"):
            choose_prompt_builder(-1)

    def test_invalid_shots_10_raises_error(self) -> None:
        """Test that shots=10 raises ValueError."""
        with pytest.raises(ValueError, match="shots must be one of"):
            choose_prompt_builder(10)

    def test_returned_function_signature(self) -> None:
        """Test that returned functions have correct signature."""
        builder_0 = choose_prompt_builder(0)
        builder_1 = choose_prompt_builder(1)
        builder_5 = choose_prompt_builder(5)

        # All should be callable
        assert callable(builder_0)
        assert callable(builder_1)
        assert callable(builder_5)

        # Test calling with sample arguments
        question = "Test question"
        headers = ["col1", "col2"]
        types = ["int", "text"]
        sample_row = [1, "test"]

        # Should not raise
        result_0 = builder_0(question, headers, types, sample_row)
        result_1 = builder_1(question, headers, types, sample_row)
        result_5 = builder_5(question, headers, types, sample_row)

        # All should return strings
        assert isinstance(result_0, str)
        assert isinstance(result_1, str)
        assert isinstance(result_5, str)

        # Basic verification that prompt contains the question
        assert "Test question" in result_0
        assert "Test question" in result_1
        assert "Test question" in result_5
