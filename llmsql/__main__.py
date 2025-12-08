from llmsql._cli import LLMSQLCLI


def main() -> None:
    """Main CLI entry point."""
    parser = LLMSQLCLI()
    args = parser.parse_args()
    parser.execute(args)


if __name__ == "__main__":
    main()
