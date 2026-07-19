"""Tests for the pure logic in main.py: extraction helpers, command parsing,
model resolution, and output formatting.

Anything that would hit the Anthropic API (agent.invoke) is intentionally not
exercised here; those paths are thin glue over LangChain.
"""
import main
from conftest import make_message


# --------------------------------------------------------------------------- #
# extract_tool_usage
# --------------------------------------------------------------------------- #
def test_extract_tool_usage_collects_unique_names_in_order():
    messages = [
        make_message(name="wikipedia_search"),
        make_message(name="multiply"),
        make_message(name="wikipedia_search"),  # duplicate ignored
        make_message(name=None),                # non-tool message ignored
    ]
    assert main.extract_tool_usage(messages) == ["wikipedia_search", "multiply"]


def test_extract_tool_usage_empty():
    assert main.extract_tool_usage([]) == []


# --------------------------------------------------------------------------- #
# extract_tool_results
# --------------------------------------------------------------------------- #
def test_extract_tool_results_returns_named_content():
    messages = [
        make_message(name="multiply", content="12"),
        make_message(name=None, content="assistant text"),   # skipped: no name
        make_message(name="ddg", content=None),              # skipped: no content
    ]
    results = main.extract_tool_results(messages)
    assert results == [{"tool": "multiply", "result": "12"}]


def test_extract_tool_results_stringifies_non_string_content():
    messages = [make_message(name="multiply", content=12)]
    assert main.extract_tool_results(messages) == [{"tool": "multiply", "result": "12"}]


# --------------------------------------------------------------------------- #
# extract_sources
# --------------------------------------------------------------------------- #
def test_extract_sources_finds_and_cleans_urls():
    messages = [
        make_message(content="See https://example.com/page, and http://foo.org."),
        make_message(content="dup https://example.com/page again"),  # de-duplicated
    ]
    sources = main.extract_sources(messages)
    # Trailing punctuation is stripped; duplicates removed; order preserved.
    assert sources == ["https://example.com/page", "http://foo.org"]


def test_extract_sources_ignores_non_urls():
    messages = [make_message(content="no links here just text")]
    assert main.extract_sources(messages) == []


def test_extract_sources_known_limitation_leading_punctuation():
    # Documents current behavior: a URL wrapped in parentheses is NOT detected,
    # because extraction only matches tokens that *start* with http(s) and only
    # strips trailing punctuation (see main.extract_sources).
    messages = [make_message(content="(https://wrapped.example/x)")]
    assert main.extract_sources(messages) == []


# --------------------------------------------------------------------------- #
# extract_token_usage
# --------------------------------------------------------------------------- #
def test_extract_token_usage_sums_across_messages():
    messages = [
        make_message(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
        make_message(usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}),
    ]
    usage = main.extract_token_usage(messages)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (12, 8, 20)


def test_extract_token_usage_falls_back_when_total_missing():
    messages = [make_message(usage_metadata={"input_tokens": 7, "output_tokens": 4})]
    usage = main.extract_token_usage(messages)
    assert usage.total_tokens == 11  # computed from input + output


def test_extract_token_usage_ignores_non_dict_metadata():
    messages = [
        make_message(usage_metadata=None),
        make_message(usage_metadata="not-a-dict"),
        make_message(),  # no usage_metadata attribute value
    ]
    usage = main.extract_token_usage(messages)
    assert usage.total_tokens == 0


# --------------------------------------------------------------------------- #
# format_tool_result
# --------------------------------------------------------------------------- #
def test_format_tool_result_pretty_prints_json():
    out = main.format_tool_result('{"b": 2, "a": 1}')
    assert '"b": 2' in out and "\n" in out  # indented multi-line


def test_format_tool_result_handles_python_literal():
    out = main.format_tool_result("[{'title': 'x'}]")
    assert '"title": "x"' in out


def test_format_tool_result_passes_through_plain_text():
    assert main.format_tool_result("just a string") == "just a string"


# --------------------------------------------------------------------------- #
# _prefix_match
# --------------------------------------------------------------------------- #
def test_prefix_match_unique():
    assert main._prefix_match("mo", ["help", "model", "show"]) == "model"


def test_prefix_match_ambiguous_returns_none():
    assert main._prefix_match("s", ["show", "sync"]) is None


def test_prefix_match_unknown_returns_none():
    assert main._prefix_match("zzz", ["help", "model"]) is None


# --------------------------------------------------------------------------- #
# expand_command
# --------------------------------------------------------------------------- #
def test_expand_command_bare_slash_is_help():
    assert main.expand_command("/") == "/help"


def test_expand_command_help_abbrev():
    assert main.expand_command("/h") == "/help"


def test_expand_command_model_with_arg():
    assert main.expand_command("/m 3") == "/model 3"


def test_expand_command_show_sources_abbrev():
    assert main.expand_command("/s s") == "/show sources"


def test_expand_command_show_tool_results_abbrev():
    assert main.expand_command("/s t") == "/show tool results"


def test_expand_command_unknown_passes_through():
    assert main.expand_command("/xyz") == "/xyz"


# --------------------------------------------------------------------------- #
# resolve_model_choice / MODEL_CHOICES
# --------------------------------------------------------------------------- #
def test_resolve_model_choice_by_number_and_name():
    assert main.resolve_model_choice("2") == ("opus", "claude-opus-4-8")
    assert main.resolve_model_choice("HAIKU") == ("haiku", "claude-haiku-4-5-20251001")


def test_resolve_model_choice_unknown():
    assert main.resolve_model_choice("gpt") is None


def test_model_choices_number_and_name_agree():
    # "1"/"haiku", "2"/"opus", "3"/"sonnet" must map to the same model id.
    for num, name in (("1", "haiku"), ("2", "opus"), ("3", "sonnet")):
        assert main.MODEL_CHOICES[num] == main.MODEL_CHOICES[name]


# --------------------------------------------------------------------------- #
# process_command
# --------------------------------------------------------------------------- #
def test_process_command_help_returns_none(capsys):
    assert main.process_command("/help", "claude-haiku-4-5-20251001") is None


def test_process_command_model_switch_returns_new_model():
    result = main.process_command("/model opus", "claude-haiku-4-5-20251001")
    assert result == "claude-opus-4-8"


def test_process_command_unknown_model_keeps_current(capsys):
    current = "claude-haiku-4-5-20251001"
    assert main.process_command("/model gpt", current) == current


def test_process_command_non_command_returns_current():
    current = "claude-sonnet-5"
    assert main.process_command("hello world", current) == current
