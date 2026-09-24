# lab-ai-agents-workbook

A workbook of standalone AI-agent experiments. Each experiment lives in its own numbered folder.

| Folder | Description |
| --- | --- |
| `11-openai-agents-sdk-bedrock/` | OpenAI Agents SDK running against Amazon Bedrock |

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

The script prints the active AWS profile and region, then sends a prompt to the agent and prints the response.

## Adding dependencies

```bash
uv add <package>
```

The project was set up with:

```bash
uv add openai-agents
uv add "openai[bedrock]"
```
