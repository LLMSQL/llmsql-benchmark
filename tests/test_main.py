# tests/test_main.py

import llmsql.__main__ as main_module


def test_main_calls_parser_and_execute(monkeypatch):
    called = {}

    class DummyParser:
        def parse_args(self):
            called["parse_args"] = True
            return "parsed-args"

        def execute(self, args):
            called["execute"] = args

    monkeypatch.setattr(main_module, "ParserCLI", DummyParser)

    main_module.main()

    assert called["parse_args"] is True
    assert called["execute"] == "parsed-args"
