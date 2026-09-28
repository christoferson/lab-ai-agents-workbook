# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A workbook of standalone AI-agent experiments. Each experiment lives in its own numbered folder (e.g. `11-openai-agents-sdk-bedrock/`) with self-contained scripts. The root `main.py` is only the `uv init` placeholder.

## Environment & Commands

- Python 3.13 (`.python-version`), managed with **uv**. All folders share one root `pyproject.toml` / `.venv`.
- Install deps: `uv sync`
- Add a dependency: `uv add <package>` (e.g. `uv add openai-agents`, `uv add "openai[bedrock]"`)
- Run an experiment: `uv run 11-openai-agents-sdk-bedrock/openai-agent.py`

There is no test suite, linter, or build step configured.

## Bedrock / OpenAI Agents SDK pattern

`11-openai-agents-sdk-bedrock/openai-agent.py` shows the setup other experiments are likely to reuse:

- Uses the OpenAI Agents SDK (`agents` package) against Amazon Bedrock rather than the OpenAI API, via `AsyncOpenAI(provider=bedrock(region=...))` from `openai.providers`.
- That client is registered globally with `set_default_openai_client(...)`, so `Agent(...)` / `Runner.run(...)` route to Bedrock.
- Tracing is turned off with `set_tracing_disabled(True)` because there is no OpenAI tracing endpoint to export to.
- Model IDs are Bedrock IDs, read from `BEDROCK_MODEL_ID` (defaults to `openai.gpt-5.6-luna`).
- AWS credentials and region come from the `AWS_PROFILE` and `AWS_REGION` environment variables, which must be set before running (for example with `uv run --env-file .env ...`).

## Claude Agent SDK pattern

`21-claude-agents-sdk-bedrock/` uses `claude-agent-sdk`, which is a different model from the OpenAI Agents SDK:

- `query(prompt=..., options=ClaudeAgentOptions(...))` spawns the Claude Code CLI bundled in the package as a subprocess and yields `AssistantMessage` / `ResultMessage` objects. There is no client object to register.
- Bedrock is selected by passing `CLAUDE_CODE_USE_BEDROCK=1` (plus `AWS_PROFILE` / `AWS_REGION`) through `options.env`.
- Model IDs are Bedrock inference profile IDs, read from `BEDROCK_CLAUDE_MODEL_ID` (defaults to `global.anthropic.claude-sonnet-5`).
- `tools=[]` disables Claude Code's built-in tools (file access, shell, and so on). Enabling them lets the agent act on the machine it runs on.
- Subagents: pass `agents={name: AgentDefinition(...)}`, enable the built-in tool with `tools=["Agent"]` plus `allowed_tools`, and set `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` in `options.env`. Without that setting, subagents run in the background, and the caller may move on without their results.
- There is no `output_type`. For structured output either put a JSON schema in the system prompt and parse the reply, or pass `SomeModel.model_json_schema()` as a `@tool` input schema and read the object off the tool call's arguments. `create_sdk_mcp_server` validates each call with `jsonschema` before the handler runs, so the second way can't yield a malformed object.
- There is no handoff primitive. `claude-agent-handoffs.py` builds one from a session: the first agent calls a tool that ends its run, then the code runs the next agent with `resume=<session id>` and its own `system_prompt` / `tools`. Resuming carries the history, not the agent, and both runs share one session ID.

## Strands Agents SDK pattern

`31-strands-agents-sdk-bedrock/` uses `strands-agents`, AWS's agent SDK:

- `Agent(model=BedrockModel(model_id=..., boto_session=boto3.Session(profile_name=..., region_name=...)), system_prompt=...)`. Calling `agent(prompt)` is synchronous and returns an `AgentResult`, and `str(result)` gives the final text.
- Always pass `model=` explicitly. A bare `Agent()` falls back to a default Bedrock model.
- Pass `callback_handler=None` unless you want the default handler to print the response as it streams.
- `STRANDS_MODEL_PROVIDER` selects `anthropic` (default), `openai` or `amazon`. All three use `BedrockModel` (the Converse API).
  - `anthropic` reads its model ID from `STRANDS_MODEL_ID_ANTHROPIC`, which defaults to `global.anthropic.claude-sonnet-5`.
  - `openai` reads `STRANDS_MODEL_ID_OPENAI`, which defaults to gpt-oss `openai.gpt-oss-120b-1:0`.
  - `amazon` reads `STRANDS_MODEL_ID_AMAZON`, which defaults to Nova 2 Lite `global.amazon.nova-2-lite-v1:0`.
- Converse rejects `openai.gpt-5.6-luna`, so don't reuse `BEDROCK_MODEL_ID` here.
