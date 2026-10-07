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
- Guardrails are hooks (`options.hooks`, keyed by event, with `HookMatcher(matcher=..., hooks=[...], timeout=...)`). A `UserPromptSubmit` hook returning `{"decision": "block", "reason": ...}` drops the prompt before the model runs; a `PreToolUse` hook returns `permissionDecision` `"allow"` / `"deny"`, and `permissionDecisionReason` reaches the agent as the tool result, so it can retry. `{"continue_": False, "stopReason": ...}` ends the run, but `ResultMessage` still says success with empty text, so track it yourself. Hooks see tool calls, not the final reply, so route anything you need to guard through a tool.
- There is no `output_type`. For structured output either put a JSON schema in the system prompt and parse the reply, or pass `SomeModel.model_json_schema()` as a `@tool` input schema and read the object off the tool call's arguments. `create_sdk_mcp_server` validates each call with `jsonschema` before the handler runs, so the second way can't yield a malformed object.
- Built-in `WebSearch` is not available on Bedrock (the agent is never offered it), but `WebFetch` works. `claude-agent-deep-research.py` makes `web_search` an MCP tool that calls Bedrock web search through the Responses API (`AsyncOpenAI(provider=bedrock(...))`, model from `BEDROCK_MODEL_ID`). That usage is not in `total_cost_usd`.
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
- Tools: decorate a plain function with `@tool` (`from strands import tool`) and pass it as `Agent(tools=[...])`. The schema comes from the type hints and the docstring (first line plus `Args:`), and `tool_spec` shows it. A tool returns a plain value. Tool calls and results are `toolUse` / `toolResult` blocks in `agent.messages`, matched by `toolUseId`.
- Conversation history lives on the `Agent`: each call appends to `agent.messages` (a plain list in Converse format) and sends it back next time. Reuse the agent to continue; make a new one to start fresh. A `BedrockModel` holds no history, so agents can share one.
- Sessions: `Agent(session_manager=FileSessionManager(session_id=..., storage_dir=...), agent_id=...)` writes each message to JSON files as it is added, and a new agent with the same IDs reloads them at construction. The system prompt isn't stored. Constructing the manager creates the session, and `delete_session` raises if it's missing. `strands-agent-session.py` stores in the git-ignored `31-strands-agents-sdk-bedrock/sessions/`.
- Parallel agents: `await agent.invoke_async(prompt)` is the async form of `agent(prompt)`, so `asyncio.gather` runs separate agents concurrently. Per-agent token usage is in `agent.event_loop_metrics.accumulated_usage`. `strands-agent-workflow.py` saves to the git-ignored `itineraries/strands/`.
- Agents as tools: `agent.as_tool(name=..., description=...)` (name must match `[a-zA-Z0-9_-]`) gives a tool with one `input` string. By default each call resets the agent to its initial messages. An `Agent` placed directly in `tools=[...]` is wrapped automatically. Parallel tool requests run concurrently. Sub-agent tokens are counted on the sub-agent, not the caller. `as_tool(delegate=True)` returns the sub-agent's reply as the caller's final answer.
- Handoffs: `Swarm([agents], entry_point=..., max_handoffs=...)` from `strands.multiagent` adds a `handoff_to_agent(agent_name, message, context)` tool to each agent. Agent names are the node IDs. The next agent does **not** get the conversation, only a prompt Swarm builds from the handoff message, the task, the node history and the shared context. So the handing-off agent must put what's needed in the message, and its steps belong in its system prompt, not the task (every agent sees the task). `max_handoffs` counts agent turns. An agent's messages are reset before each turn, but its `event_loop_metrics` aren't, so `result.accumulated_usage` overcounts agents that get control twice. Use per-agent `event_loop_metrics` instead. `result.node_history` and `result.results[name].result` describe the run.
- Structured output: `Agent(structured_output_model=SomeModel)` (or per call, `agent(prompt, structured_output_model=...)`) puts the parsed Pydantic object on `result.structured_output`. Strands adds a tool named after the class and validates its arguments with Pydantic. Errors go back to the model, and if the model ends without calling the tool, Strands forces it with `toolChoice`. `Agent.structured_output()` is deprecated.
- Guardrails are interventions: `Agent(interventions=[...])` takes `InterventionHandler` subclasses (a `name`, plus overrides of `before_invocation` / `after_model_call` / `before_tool_call` / ..., sync or async) that return `Proceed`, `Deny`, `Guide`, `Confirm` or `Transform`. `Deny` in `before_invocation` skips the model and replies `"DENIED: <reason>"` (stop reason `end_turn`, no flag), and leaves only that reply in `agent.messages`, so don't call that agent again. `Guide(feedback=...)` in `after_model_call` discards the reply and retries with the feedback as a user message, with no retry cap. `Deny` is a no-op there, so an output guardrail can't block; track the decision yourself.
- An `Agent` can't run two calls at once (a second `invoke_async` raises `ConcurrencyException`), so parallel work needs one agent per task. `strands-agent-deep-research.py` builds a fresh agent per step on one shared `BedrockModel`. Its `web_search` is an `async` `@tool` that calls Bedrock web search through the Responses API (`AsyncOpenAI(provider=bedrock(...))`, model from `BEDROCK_MODEL_ID`), without `tool_choice="required"`, which timed out in October 2026 tests. Strands has no wall-clock limit for an agent run; `asyncio.wait_for(agent.invoke_async(...), seconds)` works and cancels a pending model or tool call. A `BedrockModel(boto_client_config=...)` replaces Strands' default config (`read_timeout=120`), so set the timeout and retries in it yourself. Strands' core package has no page fetch tool like `WebFetch`. Reports go to the git-ignored `reports/strands/`.
