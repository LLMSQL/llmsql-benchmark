from abc import ABC, abstractmethod
import argparse
from typing import Any, TypeVar

T = TypeVar("T", bound="SubCommand")


class SubCommand(ABC):
    """Base class for all subcommands."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Subclasses must implement their own initializer."""
        pass

    @classmethod
    def create(cls: type[T], subparsers: argparse._SubParsersAction) -> T:
        """Factory method to create and register a command instance."""
        return cls(subparsers)

    @abstractmethod
    def _add_args(self) -> None:
        """Add arguments specific to this subcommand."""
        pass
