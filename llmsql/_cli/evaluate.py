import argparse
from typing import Any

import yaml

from llmsql._cli.subparsers import SubCommand
from llmsql.config.config import DEFAULT_LLMSQL_VERSION
from llmsql.evaluation.evaluate import evaluate


class Evaluate(SubCommand):
    """Command for LLM evaluation"""

    def __init__(
        self, subparsers: argparse._SubParsersAction, *args: Any, **kwargs: Any
    ) -> None:
        self._parser = subparsers.add_parser(
            "evaluate",
            help="Evaluate predictions against the LLMSQL benchmark",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        self._add_args()
        self._parser.set_defaults(func=self._execute)

    def _add_args(self) -> None:
        """Add evaluation-specific arguments to the parser."""
        self._parser.add_argument(
            "--outputs",
            type=str,
            required=True,
            help="Path to the .json file containing the model's generated queries.",
        )

        self._parser.add_argument(
            "--version",
            type=str,
            default=DEFAULT_LLMSQL_VERSION,
            choices=["1.0", "2.0"],
            help=f"LLMSQL benchmark version (default:{DEFAULT_LLMSQL_VERSION})",
        )

        self._parser.add_argument(
            "--workdir-path",
            help="Directory for benchmark downloads. If omitted, a temporary directory is used.",
        )

        self._parser.add_argument(
            "--show-mismatches",
            action="store_true",
            default=True,
            help="Print SQL mismatches during evaluation",
        )

        self._parser.add_argument(
            "--max-mismatches",
            type=int,
            default=5,
            help="Maximum mismatches to display",
        )

        self._parser.add_argument(
            "--save-report",
            type=str,
            default=None,
            help="Path to save evaluation report JSON.",
        )

        self._parser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="Name of the evaluated model (stored in the report and YAML).",
        )

        self._parser.add_argument(
            "--save-leaderboard-yaml",
            type=str,
            default=None,
            help=(
                "Path to additionally save results in the leaderboard run.yaml "
                "format (see the leaderboard/ folder)."
            ),
        )

        self._parser.add_argument(
            "--run-metadata",
            type=str,
            default=None,
            help=(
                "Path to a YAML/JSON file with extra run metadata (model, type, "
                "inference backend/arguments, ...) merged into the leaderboard YAML."
            ),
        )

    @staticmethod
    def _execute(args: argparse.Namespace) -> None:
        """Execute the evaluate function with parsed arguments."""
        try:
            run_metadata = None
            if args.run_metadata:
                with open(args.run_metadata, encoding="utf-8") as f:
                    run_metadata = yaml.safe_load(f) or {}
                if not isinstance(run_metadata, dict):
                    raise ValueError("--run-metadata file must contain a mapping")

            evaluate(
                outputs=args.outputs,
                version=args.version,
                workdir_path=args.workdir_path,
                save_report=args.save_report,
                show_mismatches=args.show_mismatches,
                max_mismatches=args.max_mismatches,
                model_name=args.model_name,
                save_leaderboard_yaml=args.save_leaderboard_yaml,
                run_metadata=run_metadata,
            )
        except Exception as e:
            print(f"Error during evaluation: {e}")
