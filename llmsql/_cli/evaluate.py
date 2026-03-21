import argparse
from typing import Any

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

    @staticmethod
    def _execute(args: argparse.Namespace) -> None:
        """Execute the evaluate function with parsed arguments."""
        try:
            evaluate(
                outputs=args.outputs,
                version=args.version,
                workdir_path=args.workdir_path,
                save_report=args.save_report,
                show_mismatches=args.show_mismatches,
                max_mismatches=args.max_mismatches,
            )
        except Exception as e:
            print(f"Error during evaluation: {e}")
