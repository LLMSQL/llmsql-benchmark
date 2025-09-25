import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmsql", description="LLMSQL CLI")
    subparsers = parser.add_subparsers(dest="command")

    ft_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune a causal LM on the LLMSQL benchmark.",
        description="Launch fine-tuning using Hugging Face TRL's SFTTrainer.\n\n"
        "You can pass parameters directly on the CLI or provide a YAML "
        "config file via --config-file. CLI args take precedence.",
        epilog="Example:\n  llmsql finetune --config-file examples/example_finetune_args.yaml",
    )
    ft_parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to a YAML config file containing training parameters.",
    )

    args, extra = parser.parse_known_args()

    if args.command == "finetune":
        from llmsql.finetune import finetune

        sys.argv = ["llmsql-finetune"] + extra + ["--config_file", args.config_file]
        finetune.run_cli()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
