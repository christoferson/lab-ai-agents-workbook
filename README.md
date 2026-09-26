# lab-ai-agents-workbook

A workbook of standalone AI-agent experiments. Each experiment lives in its own numbered folder with its own README.

| Folder | Description |
| --- | --- |
| [`11-openai-agents-sdk-bedrock/`](11-openai-agents-sdk-bedrock/README.md) | OpenAI Agents SDK running against Amazon Bedrock |
| [`21-claude-agents-sdk-bedrock/`](21-claude-agents-sdk-bedrock/README.md) | Claude Agent SDK running against Amazon Bedrock |

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

`uv sync` creates `.venv/` and installs the dependencies pinned in `uv.lock`. All folders share this one environment.

## Configure AWS

The scripts read the AWS profile and region from environment variables. Create a `.env` file in the repo root:

```dotenv
AWS_PROFILE=your-profile
AWS_REGION=us-east-1
# Optional: model for the OpenAI Agents SDK examples, defaults to openai.gpt-5.6-luna
BEDROCK_MODEL_ID=openai.gpt-5.6-luna
# Optional: model for the Claude Agent SDK examples, defaults to global.anthropic.claude-sonnet-5
BEDROCK_CLAUDE_MODEL_ID=global.anthropic.claude-sonnet-5
```

Alternatively, export them in your shell:

```bash
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1
```

## Run

Run scripts from the repo root:

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

See each folder's README for its scripts and what they demonstrate.

## Adding dependencies

```bash
uv add <package>
```

The project was set up with:

```bash
uv add openai-agents
uv add "openai[bedrock]"
uv add rich
uv add claude-agent-sdk
```
