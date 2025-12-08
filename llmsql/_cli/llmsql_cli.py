import argparse
import textwrap
from typing import Any

from llmsql._cli.evaluation import EvaluationCommand
from llmsql._cli.inference import InferenceCommand


class LLMSQLCLI:
    """Top-level LLMSQL CLI orchestrator."""

    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser(
            prog="llmsql",
            description="LLMSQL: LLM-powered SQL generation toolkit",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
                Top-level commands:
                  inference    Run SQL generation using a chosen LLM backend
                  evaluate     Evaluate predictions against the LLMSQL benchmark
            """),
        )

        self._parser.set_defaults(func=lambda args: self._parser.print_help())
        self._subparsers = self._parser.add_subparsers(
            dest="command", metavar="COMMAND"
        )

        # Register subcommands
        InferenceCommand.register(self._subparsers)
        EvaluationCommand.register(self._subparsers)

    # ----------------------------------------------------------------------
    def parse_args(self) -> argparse.Namespace:
        """Parse arguments"""
        return self._parser.parse_args()

    # ----------------------------------------------------------------------
    def execute(self, args: argparse.Namespace) -> Any:
        """Execute a chosen subcommand."""
        return args.func(args)
