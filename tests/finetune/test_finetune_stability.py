import json
from pathlib import Path

import pytest

import llmsql.finetune.finetune as finetune


@pytest.mark.asyncio
async def test_parse_args_and_config_with_yaml(tmp_path, monkeypatch):
    # create a fake yaml config file
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        """
        model_name_or_path: "gpt2"
        output_dir: "out"
        num_train_epochs: 2
        train_file: "train.jsonl"
        val_file: "val.jsonl"
        """
    )

    testargs = ["prog", "--config_file", str(yaml_file)]
    monkeypatch.setattr("sys.argv", testargs)

    args = finetune.parse_args_and_config()
    assert args.model_name_or_path == "gpt2"
    assert args.output_dir == "out"
    assert args.num_train_epochs == 2
    assert args.train_file == "train.jsonl"
    assert args.val_file == "val.jsonl"


@pytest.mark.asyncio
async def test_build_dataset(monkeypatch, tmp_path):
    # fake tables
    tables = {
        "t1": {
            "table_id": "t1",
            "header": ["col1"],
            "types": ["text"],
            "rows": [["foo"]],
        }
    }
    # fake questions
    q_file = tmp_path / "train.jsonl"
    q_file.write_text(
        json.dumps({"question": "What?", "sql": "SELECT 1", "table_id": "t1"}) + "\n"
    )

    # fake prompt builder
    def fake_builder(question, header, types, example_row):
        return f"{question} | {header} | {types} | {example_row}"

    monkeypatch.setattr(
        "llmsql.finetune.finetune.load_jsonl",
        lambda f: [json.loads(line) for line in open(f)],
    )

    dataset = finetune.build_dataset(str(q_file), tables, fake_builder)
    assert len(dataset) == 1
    sample = dataset[0]
    assert "prompt" in sample and "completion" in sample
    assert sample["completion"] == "SELECT 1"


@pytest.mark.asyncio
async def test_download_file(monkeypatch, tmp_path):
    def fake_download(**kwargs):
        file_path = tmp_path / kwargs["filename"]
        file_path.write_text("content")
        return str(file_path)

    monkeypatch.setattr("llmsql.finetune.finetune.hf_hub_download", fake_download)
    monkeypatch.setattr("shutil.copy", lambda src, dst: Path(dst).write_text("copied"))

    out = finetune._download_file("file.txt", "repo", str(tmp_path))
    assert Path(out).exists()


@pytest.mark.asyncio
async def test_main_runs_with_mocks(tmp_path, monkeypatch):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    tables_file = tmp_path / "tables.jsonl"

    train_file.write_text(
        json.dumps({"question": "Q?", "sql": "SELECT 1", "table_id": "t1"}) + "\n"
    )
    val_file.write_text(
        json.dumps({"question": "Q?", "sql": "SELECT 2", "table_id": "t1"}) + "\n"
    )
    tables_file.write_text(
        json.dumps(
            {"table_id": "t1", "header": ["c"], "types": ["text"], "rows": [["r"]]}
        )
        + "\n"
    )

    monkeypatch.setattr(
        "llmsql.finetune.finetune.load_jsonl",
        lambda f: [json.loads(line) for line in open(f)],
    )
    monkeypatch.setattr(
        "llmsql.finetune.finetune.choose_prompt_builder",
        lambda shots: lambda q, h, t, r: "PROMPT",
    )
    monkeypatch.setattr(
        "llmsql.finetune.finetune.AutoModelForCausalLM",
        type("FakeModel", (), {"from_pretrained": lambda *a, **k: "MODEL"}),
    )
    monkeypatch.setattr(
        "llmsql.finetune.finetune.SFTConfig", lambda **kwargs: {"args": kwargs}
    )

    class FakeTrainer:
        def __init__(self, model, train_dataset, eval_dataset, args):
            pass

        def train(self):
            return "trained"

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("llmsql.finetune.finetune.SFTTrainer", FakeTrainer)

    finetune.main(
        model_name_or_path="gpt2",
        output_dir=str(tmp_path / "out"),
        train_file=str(train_file),
        val_file=str(val_file),
        tables_file=str(tables_file),
        shots=1,
        num_train_epochs=1,
    )

    assert (tmp_path / "out" / "final_model").exists()


@pytest.mark.asyncio
async def test_flatten_nested_config():
    """Test that nested YAML config is flattened correctly."""
    from llmsql.finetune.finetune import parse_args_and_config

    # Create a config with nested structure
    config = {
        "training": {"num_epochs": 10, "learning_rate": 0.001},
        "model": {"name": "gpt2", "size": "small"},
        "output_dir": "test_out",
    }

    import sys
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    old_argv = sys.argv
    try:
        sys.argv = ["prog", "--config_file", config_path]
        args = parse_args_and_config()

        # Flattened keys should use underscore separator
        assert args.training_num_epochs == 10
        assert args.training_learning_rate == 0.001
        assert args.model_name == "gpt2"
        assert args.model_size == "small"
        assert args.output_dir == "test_out"
    finally:
        sys.argv = old_argv
        import os

        os.unlink(config_path)


@pytest.mark.asyncio
async def test_main_with_wandb_env_vars(tmp_path, monkeypatch):
    """Test that WandB environment variables are set correctly."""
    import os

    # Clear any existing WandB env vars
    for key in [
        "WANDB_PROJECT",
        "WANDB_RUN_NAME",
        "WANDB_API_KEY",
        "WANDB_MODE",
    ]:
        if key in os.environ:
            del os.environ[key]

    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    tables_file = tmp_path / "tables.jsonl"

    train_file.write_text(
        json.dumps({"question": "Q?", "sql": "SELECT 1", "table_id": "t1"}) + "\n"
    )
    val_file.write_text(
        json.dumps({"question": "Q?", "sql": "SELECT 2", "table_id": "t1"}) + "\n"
    )
    tables_file.write_text(
        json.dumps(
            {"table_id": "t1", "header": ["c"], "types": ["text"], "rows": [["r"]]}
        )
        + "\n"
    )

    # Mock all the heavy dependencies
    monkeypatch.setattr(
        "llmsql.finetune.finetune.load_jsonl",
        lambda f: [json.loads(line) for line in open(f)],
    )
    monkeypatch.setattr(
        "llmsql.finetune.finetune.choose_prompt_builder",
        lambda shots: lambda q, h, t, r: "PROMPT",
    )
    monkeypatch.setattr(
        "llmsql.finetune.finetune.AutoModelForCausalLM",
        type("FakeModel", (), {"from_pretrained": lambda *a, **k: "MODEL"}),
    )
    monkeypatch.setattr(
        "llmsql.finetune.finetune.SFTConfig", lambda **kwargs: {"args": kwargs}
    )

    class FakeTrainer:
        def __init__(self, model, train_dataset, eval_dataset, args):
            pass

        def train(self):
            return "trained"

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("llmsql.finetune.finetune.SFTTrainer", FakeTrainer)

    # Call main with WandB parameters
    finetune.main(
        model_name_or_path="gpt2",
        output_dir=str(tmp_path / "out"),
        train_file=str(train_file),
        val_file=str(val_file),
        tables_file=str(tables_file),
        shots=1,
        num_train_epochs=1,
        wandb_project="test_project",
        wandb_run_name="test_run",
        wandb_key="test_key",
        wandb_offline=True,
    )

    # Verify environment variables were set
    assert os.environ.get("WANDB_PROJECT") == "test_project"
    assert os.environ.get("WANDB_RUN_NAME") == "test_run"
    assert os.environ.get("WANDB_API_KEY") == "test_key"
    assert os.environ.get("WANDB_MODE") == "offline"
