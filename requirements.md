# AgentTutorial Features

## Overview
This project is a terminal-based LangChain research assistant that uses Anthropic chat models, supports conversation memory, and can call multiple tools for research and web retrieval.

## Core Chat Features
- Interactive terminal chat loop in `main.py`
- Multi-turn conversation memory
- Follow-up questions reuse prior conversation context
- Human-readable output format
- Per-turn token usage display:
  - input tokens
  - output tokens
  - total tokens

## Model Switching
The chat supports runtime model switching with slash commands.

### Available models
1. `claude-haiku-4-5-20251001`
2. `claude-opus-4-8`
3. `claude-sonnet-5`

### Model commands
- `/model`
- `/model 1`
- `/model 2`
- `/model 3`
- `/model haiku`
- `/model opus`
- `/model sonnet`

### Model UX
- The current model is shown in the prompt, e.g. `[haiku] >`
- `/model` displays the current model and all available options

## Slash Commands
### Help
- `/help`
- `/`

### Command abbreviation support
Commands support unique-prefix matching.

Examples:
- `/h` → `/help`
- `/m` → `/model`
- `/m 3` → `/model 3`
- `/s s` → `/show sources`
- `/s t` → `/show tool results`
- `/s h` → `/show history`

## Tooling
The agent can call the following tools:

### 1. Wikipedia search
- Tool name: `wikipedia_search`
- Searches Wikipedia and returns a summary
- Uses structured Pydantic tool inputs

### 2. Multiply
- Tool name: `multiply`
- Multiplies two numbers
- Uses structured Pydantic tool inputs

### 3. DuckDuckGo search
- Tool name: `duckduckgo_search`
- Runs web searches using DuckDuckGo
- Returns structured search results with:
  - title
  - url
  - snippet
- Uses the `ddgs` package

### 4. Web page fetch
- Tool name: `fetch_webpage`
- Fetches a web page by URL
- Extracts readable text content from HTML
- Strips noisy HTML elements such as scripts/styles/navigation areas
- Supports configurable maximum returned text length

## Tool Tracing
Each tool prints a visible trace when called.

Tool trace behavior includes:
- tool name
- invocation details
- preview of the first result line
- error trace if the tool fails

## Tool Result Memory
The application stores tool results from the most recent query in memory.

### Show tool results
- `/show tool results`
- abbreviation: `/s t`

Behavior:
- shows the full stored tool outputs for the last query
- formats structured outputs more readably
- groups outputs by tool

## Source Memory
The application stores extracted source URLs from the most recent query in memory.

### Show sources
- `/show sources`
- abbreviation: `/s s`

Behavior:
- sources are not shown automatically in normal answer output
- sources can be displayed on demand for the last query only

## Query History
The application keeps an in-session history of every query.

Each history entry records:
- the original query text
- a single-line summary of the answer (the output)
- the token cost of that query:
  - input tokens
  - output tokens
  - total tokens

### Show history
- `/show history`
- abbreviation: `/s h`

Behavior:
- lists every query made during the current session, in order
- shows the per-query summary and token cost
- shows the total cost (input, output, total tokens) across all queries
- history is kept in memory for the current session only

## Output Behavior
Normal answer output includes:
- Answer
- Tools used
- Token usage

Normal answer output does not automatically include:
- full tool outputs
- source URLs

These are available via slash commands instead.

## Data Modeling
Structured data is used in the tool layer with Pydantic models for:
- Wikipedia tool input
- Multiply tool input
- DuckDuckGo search input
- Web page fetch input

## Dependencies
The project currently relies on:
- `langchain`
- `langchain-anthropic`
- `langchain-community`
- `langchain-openai`
- `langchain-classic`
- `python-dotenv`
- `pydantic`
- `wikipedia`
- `ddgs`
- `httpx`
- `beautifulsoup4`

## Exit Commands
- `exit`
- `quit`
