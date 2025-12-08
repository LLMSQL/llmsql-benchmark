import argparse
import json

from llmsql.config.config import DEFAULT_WORKDIR_PATH
from llmsql.evaluation.evaluate import evaluate


class EvaluationCommand:
    """CLI wrapper for the `evaluate()` function."""

    @staticmethod
    def register(subparsers: argparse._SubParsersAction) -> None:
        eval_parser = subparsers.add_parser(
            "evaluate",
            help="Evaluate SQL predictions against the LLMSQL benchmark.",
            description="Evaluate predicted SQL against the LLMSQL benchmark.",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        eval_parser.add_argument(
            "--outputs",
            required=True,
            help=(
                "Path to predictions JSONL file OR inline JSON list.\n"
                "Examples:\n"
                "  --outputs outputs/preds.jsonl\n"
                '  --outputs \'[{"id":1,"sql":"SELECT ..."}]\''
            ),
        )

        eval_parser.add_argument(
            "--workdir-path",
            type=str,
            default=None,
            help=(
                f"Optional. Where to store help .db files. Default: {DEFAULT_WORKDIR_PATH}"
            ),
        )

        eval_parser.add_argument(
            "--questions-path",
            type=str,
            default=None,
            help=(
                "Optional. Where is the questions of the benchmark stored. If not provided, questions will be downloaded."
            ),
        )

        eval_parser.add_argument(
            "--db-path",
            type=str,
            default=None,
            help=(
                "Optional. Where is the db file of the benchmark stored. If not provided, db file will be downloaded."
            ),
        )

        eval_parser.add_argument(
            "--save-report",
            type=str,
            default=None,
            help=(
                "Optional. Manual save path. If None → auto-generated with name 'evaluation_results_{uuid}.json'."
            ),
        )

        # Boolean toggle
        eval_parser.add_argument(
            "--show-mismatches",
            type=bool,
            default=True,
            help="Optional. Show mismatches during evaluation. Default: True.",
        )

        eval_parser.add_argument(
            "--max-mismatches",
            type=int,
            default=5,
            help="Optional. Number of mismatches to print. Default: 5",
        )

        # Dispatcher
        eval_parser.set_defaults(func=EvaluationCommand.execute)

    # ----------------------------------------------------------
    @staticmethod
    def execute(args: argparse.Namespace) -> None:
        """Run evaluation function."""
        # Try inline JSON first; fallback to path
        try:
            parsed = json.loads(args.outputs)
            outputs = parsed if isinstance(parsed, list) else args.outputs
        except Exception:
            outputs = args.outputs

        result = evaluate(
            outputs=outputs,
            workdir_path=args.workdir_path,
            questions_path=args.questions_path,
            db_path=args.db_path,
            save_report=args.save_report,
            show_mismatches=args.show_mismatches,
            max_mismatches=args.max_mismatches,
        )

        print(json.dumps(result, indent=2))
