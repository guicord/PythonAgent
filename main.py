import ast
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from tools import TOOLS


load_dotenv()  # Load environment variables from .env file


MODEL_CHOICES = {
    "1": ("haiku", "claude-haiku-4-5-20251001"),
    "2": ("opus", "claude-opus-4-8"),
    "3": ("sonnet", "claude-sonnet-5"),
    "haiku": ("haiku", "claude-haiku-4-5-20251001"),
    "opus": ("opus", "claude-opus-4-8"),
    "sonnet": ("sonnet", "claude-sonnet-5"),
}


class StructuredResearchResponse(BaseModel):
    question: str = Field(description="Original user question")
    answer: str = Field(description="Final answer")
    key_points: list[str] = Field(default_factory=list, description="Important bullet points")
    sources: list[str] = Field(default_factory=list, description="Source URLs or source names")
    tools_used: list[str] = Field(default_factory=list, description="Tool names used to answer")


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def create_llm(model_name: str) -> ChatAnthropic:
    return ChatAnthropic(
        model=model_name,
        max_tokens=1000,
    )


def build_agent(llm: ChatAnthropic):
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=(
            "You are a helpful research assistant. "
            "Use tools whenever they help produce a better answer. "
            "When you use web tools, include source URLs in your answer. "
            "Answer with as much details as possible"
        ),
    )


def extract_tool_usage(messages: list[object]) -> list[str]:
    tools_used: list[str] = []
    for message in messages:
        name = getattr(message, "name", None)
        if name and name not in tools_used:
            tools_used.append(name)
    return tools_used


def extract_tool_results(messages: list[object]) -> list[dict[str, str]]:
    """Return full content of every tool response message."""
    results: list[dict[str, str]] = []
    for message in messages:
        name = getattr(message, "name", None)
        content = getattr(message, "content", None)
        if name and content is not None:
            results.append({"tool": name, "result": str(content)})
    return results


def extract_sources(messages: list[object]) -> list[str]:
    sources: list[str] = []
    for message in messages:
        content = str(getattr(message, "content", ""))
        for token in content.split():
            if token.startswith("http://") or token.startswith("https://"):
                cleaned = token.rstrip(").,\"]'")
                if cleaned not in sources:
                    sources.append(cleaned)
    return sources


def extract_token_usage(messages: list[object]) -> TokenUsage:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        if not isinstance(usage, dict):
            continue

        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
        total_tokens += int(usage.get("total_tokens", 0) or 0)

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


def print_human_readable(
    answer: str,
    tools_used: list[str],
    usage: TokenUsage,
) -> None:
    print("\nAnswer:")
    print(answer)

    if tools_used:
        print("\nTools used:")
        print(", ".join(tools_used))

    print("\nToken usage:")
    print(f"- Input tokens: {usage.input_tokens}")
    print(f"- Output tokens: {usage.output_tokens}")
    print(f"- Total tokens: {usage.total_tokens}")


def format_tool_result(result: str) -> str:
    try:
        parsed = json.loads(result)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(result)
        if isinstance(parsed, (list, dict, tuple)):
            return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return result


def print_tool_results(entries: list[dict[str, str]]) -> None:
    if not entries:
        print("No tool results from the last query.")
        return

    print(f"\nTool results from last query ({len(entries)}):")
    for index, entry in enumerate(entries, start=1):
        tool_name = entry["tool"].replace("_", " ").title()
        print(f"\n{'=' * 60}")
        print(f"{index}. {tool_name}")
        print(f"{'-' * 60}")
        print(format_tool_result(entry["result"]))


def print_help() -> None:
    print("\nAvailable commands (unique prefix abbreviations work, e.g. /h, /m, /s s, /s t):")
    print("- /help                 Show this help message")
    print("- /model                Show available models and choose one")
    print("- /model 1              Switch to Haiku")
    print("- /model 2              Switch to Opus")
    print("- /model 3              Switch to Sonnet")
    print("- /model haiku          Switch to Haiku")
    print("- /model opus           Switch to Opus")
    print("- /model sonnet         Switch to Sonnet")
    print("- /show sources         Show sources from the last query  (abbrev: /s s)")
    print("- /show tool results    Show full tool results from the last query  (abbrev: /s t)")
    print("- exit or quit          Leave the chat")


def print_models(current_model: str) -> None:
    alias = next(
        (a for a, m in MODEL_CHOICES.values() if m == current_model), current_model
    )
    print(f"\nCurrent model: {alias} ({current_model})")
    print("\nAvailable models:")
    print(
        f"1. haiku - claude-haiku-4-5-20251001"
        f"{' (current)' if current_model == 'claude-haiku-4-5-20251001' else ''}"
    )
    print(
        f"2. opus - claude-opus-4-8"
        f"{' (current)' if current_model == 'claude-opus-4-8' else ''}"
    )
    print(
        f"3. sonnet - claude-sonnet-5"
        f"{' (current)' if current_model == 'claude-sonnet-5' else ''}"
    )


# Top-level slash commands and their unique-prefix-matchable keys.
_TOP_COMMANDS = ["help", "model", "show"]
# /show subcommands: keyed by first word so prefix matching is unambiguous.
_SHOW_SUBCOMMANDS = [("sources", "sources"), ("tool", "tool results")]


def _prefix_match(token: str, candidates: list[str]) -> str | None:
    """Return the single candidate that starts with `token`, or None if 0 or 2+ match."""
    matches = [c for c in candidates if c.startswith(token)]
    return matches[0] if len(matches) == 1 else None


def expand_command(raw: str) -> str:
    """Expand an abbreviated slash command to its canonical full form.

    Examples::
        /h          -> /help
        /m 1        -> /model 1
        /s s        -> /show sources
        /s t        -> /show tool results
    """
    parts = raw.strip().split(maxsplit=1)
    cmd_token = parts[0].lstrip("/").lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd_token == "":
        return "/help"

    full_cmd = _prefix_match(cmd_token, _TOP_COMMANDS)
    if full_cmd is None:
        return raw  # ambiguous or unknown — pass through unchanged

    if full_cmd == "show":
        if rest:
            # Match only the first word of rest against subcommand keys.
            sub_token = rest.split()[0].lower()
            match = next(
                (full for key, full in _SHOW_SUBCOMMANDS if key.startswith(sub_token)),
                None,
            )
            if match:
                return f"/show {match}"
        return f"/show {rest}" if rest else "/show"

    return f"/{full_cmd} {rest}".strip() if rest else f"/{full_cmd}"


def resolve_model_choice(choice: str) -> tuple[str, str] | None:
    return MODEL_CHOICES.get(choice.strip().lower())


def handle_model_command(current_model: str) -> str | None:
    print_models(current_model)
    selection = input("Select a model (1/2/3/name), or press Enter to cancel: ").strip()
    if not selection:
        print("Model change cancelled.")
        return None

    resolved = resolve_model_choice(selection)
    if not resolved:
        print("Unknown model selection.")
        return None

    alias, model_name = resolved
    print(f"Switched to {alias} ({model_name}).")
    return model_name


def process_command(content: str, current_model: str) -> str | None:
    if content == "/help":
        print_help()
        return None

    if not content.startswith("/model"):
        return current_model

    parts = content.split(maxsplit=1)
    if len(parts) == 1:
        selected_model = handle_model_command(current_model)
        return selected_model or current_model

    resolved = resolve_model_choice(parts[1])
    if not resolved:
        print("Unknown model selection. Use /model or /help.")
        return current_model

    alias, model_name = resolved
    print(f"Switched to {alias} ({model_name}).")
    return model_name


def run_chat() -> None:
    current_model = "claude-haiku-4-5-20251001"
    llm = create_llm(current_model)
    agent = build_agent(llm)
    conversation: list[dict[str, str]] = []
    last_sources: list[str] = []
    last_tool_results: list[dict[str, str]] = []

    print("Enter your question (type 'exit' to quit, /help for commands):")
    while True:
        alias = next(
            (a for a, m in MODEL_CHOICES.values() if m == current_model), current_model
        )
        content = input(f"[{alias}] > ").strip()
        if not content:
            continue
        if content.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if content.startswith("/"):
                content = expand_command(content)
                cmd = content.lower()

                if cmd == "/show sources":
                    if not last_sources:
                        print("No sources from the last query.")
                    else:
                        print("\nSources from last query:")
                        for source in last_sources:
                            print(f"- {source}")
                    continue

                if cmd == "/show tool results":
                    print_tool_results(last_tool_results)
                    continue

                updated_model = process_command(content, current_model)
                if updated_model is None:
                    continue
                if updated_model != current_model:
                    current_model = updated_model
                    llm = create_llm(current_model)
                    agent = build_agent(llm)
                continue

        current_messages = [
            *conversation,
            {
                "role": "user",
                "content": content,
            },
        ]

        result = agent.invoke({"messages": current_messages})
        messages = result.get("messages", [])

        final_answer = str(messages[-1].content) if messages else ""
        tools_used = extract_tool_usage(messages)
        last_sources = extract_sources(messages)
        last_tool_results = extract_tool_results(messages)
        token_usage = extract_token_usage(messages)
        print_human_readable(final_answer, tools_used, token_usage)

        conversation.append({"role": "user", "content": content})
        conversation.append({"role": "assistant", "content": final_answer})


if __name__ == "__main__":
    run_chat()
