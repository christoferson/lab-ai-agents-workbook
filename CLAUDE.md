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
- Model IDs are Bedrock IDs (e.g. `openai.gpt-5.6-luna`).
- AWS credentials and region come from the `AWS_PROFILE` and `AWS_REGION` environment variables, which must be set before running (for example with `uv run --env-file .env ...`).
