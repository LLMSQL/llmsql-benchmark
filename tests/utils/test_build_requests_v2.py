"""Request building for LLMSQL 2.0 questions (and unchanged 1.0 behaviour)."""

from llmsql.utils.utils import (
    build_all_requests,
    build_question_prompt,
    choose_prompt_builder,
)


def _never_called(*args):
    raise AssertionError("the few-shot builder must not be used for 2.0 questions")


def test_v2_questions_use_dataset_prompt(llmsql2_questions, llmsql2_tables):
    prompts = build_all_requests(llmsql2_questions, llmsql2_tables, _never_called)
    assert prompts == [q["prompt"] for q in llmsql2_questions]


def test_v2_questions_with_chat_template(llmsql2_questions, llmsql2_tables):
    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert len(messages) == 1 and messages[0]["role"] == "user"
            return "<user>" + messages[0]["content"] + "<assistant>"

    prompts = build_all_requests(
        llmsql2_questions[:2], llmsql2_tables, _never_called, tokenizer=Tok()
    )
    assert prompts == [
        "<user>" + q["prompt"] + "<assistant>" for q in llmsql2_questions[:2]
    ]


def test_v1_questions_unchanged():
    tables = {
        "t1": {
            "table_id": "t1",
            "header": ["id", "name"],
            "types": ["real", "text"],
            "rows": [[1, "Alice"]],
        }
    }
    q = {"question_id": 1, "question": "Who?", "table_id": "t1"}
    builder = choose_prompt_builder(5)
    expected = builder("Who?", ["id", "name"], ["real", "text"], [1, "Alice"])
    assert build_question_prompt(q, tables, builder) == expected
    assert build_all_requests([q], tables, builder) == [expected]
