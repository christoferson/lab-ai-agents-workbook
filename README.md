# lab-ai-agents-workbook

A workbook of standalone AI-agent experiments. Each experiment lives in its own numbered folder.

| Folder | Description |
| --- | --- |
| `11-openai-agents-sdk-bedrock/` | OpenAI Agents SDK running against Amazon Bedrock |
| `21-claude-agents-sdk-bedrock/` | Claude Agent SDK running against Amazon Bedrock |

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (uv installs Python 3.13 automatically if it's missing)
- An AWS account with Amazon Bedrock access, with the models you want enabled in your region
- AWS credentials configured locally, e.g. via `aws configure --profile <name>` or `aws configure sso`

## Setup

```bash
git clone <repo-url>
cd lab-ai-agents-workbook
uv sync
```

`uv sync` creates `.venv/` and installs the dependencies pinned in `uv.lock`.

## Configure AWS

The scripts read the AWS profile and region from environment variables. Create a `.env` file in the repo root:

```dotenv
AWS_PROFILE=your-profile
AWS_REGION=us-east-1
# Optional: model for the OpenAI Agents SDK examples, defaults to openai.gpt-5.6-luna
BEDROCK_MODEL_ID=openai.gpt-5.6-luna
# Optional: model for the Claude Agent SDK examples, defaults to global.anthropic.claude-sonnet-5
CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5
```

Alternatively, export them in your shell:

```bash
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1
```

## Run

```bash
# with a .env file
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent.py

# or with variables already exported
uv run 11-openai-agents-sdk-bedrock/openai-agent.py
```

`uv run` does not load `.env` automatically. To skip the `--env-file` flag, set `UV_ENV_FILE` once in your shell:

```bash
export UV_ENV_FILE=.env
uv run 11-openai-agents-sdk-bedrock/openai-agent.py
```

The script prints the active AWS profile and region, then sends a prompt to the agent and prints the response, followed by the full run result (items, raw responses, usage) pretty-printed with [rich](https://github.com/Textualize/rich).

To see the response stream token by token instead, run `openai-agent-streaming.py`:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-streaming.py
```

To see tool calling, run `openai-agent-tools.py`. A "Finance Assistant" agent compares buying a car with a loan against saving up for it. It calls local Python tools (a loan payment calculator and a savings goal calculator) instead of doing the math itself, and prints each tool call and result before its answer:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-tools.py
```

To see conversation history, run `openai-agent-conversation.py`. You tell a travel planner about a relaxed, nature-focused trip to Tokyo, then ask "What should I do on my first morning?" twice: once as a fresh run, which has to ask where you're going, and once with the previous turn passed back via `to_input_list()`, which gives a tailored suggestion:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-conversation.py
```

To have the SDK manage history for you, run `openai-agent-session.py`. It passes a `SQLiteSession` to `Runner.run(..., session=session)`, which loads and saves each turn automatically. The history is stored in `sessions.db` next to the script (git-ignored), and the demo closes and reopens the session to show the conversation survives a restart. No extra dependency is needed, since it uses Python's built-in `sqlite3`:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-session.py
```

To see a multi-agent workflow orchestrated by code, run `openai-agent-workflow.py`. Three planner agents (nature, culture, slow travel) draft a Tokyo day plan in parallel with `asyncio.gather`, an editor agent picks the best one, and a publisher agent saves it to `11-openai-agents-sdk-bedrock/itineraries/` (git-ignored) using a `save_itinerary` tool:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-workflow.py
```

To let an LLM do the orchestration instead, run `openai-agent-agents-as-tools.py`. The same three planners are wrapped with `agent.as_tool(...)` and given, together with `save_itinerary`, to a "Trip Director" planning agent. The director decides on its own to call each planner, compare the drafts and save the winner, and the script prints each call it made:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-agents-as-tools.py
```

### Claude Agent SDK

`claude-agent.py` asks a Claude model on Bedrock a single question and prints the answer plus a run summary (turns, duration, cost). The `claude-agent-sdk` package bundles the Claude Code CLI and runs it in the background, so there's nothing else to install. The script sets `CLAUDE_CODE_USE_BEDROCK=1` so the CLI uses your AWS credentials instead of an Anthropic API key:

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent.py
```

## Adding dependencies

```bash
uv add <package>
```

The project was set up with:

```bash
uv add openai-agents
uv add "openai[bedrock]"
```
