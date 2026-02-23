from llmsql._cli import ParserCLI


def main() -> None:
    """Main CLI entry point."""
    parser = ParserCLI()
    args = parser.parse_args()
    parser.execute(args)


if __name__ == "__main__":
    main()
