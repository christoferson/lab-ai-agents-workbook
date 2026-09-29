# Claude Agent SDK on Amazon Bedrock

Examples of the [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview) (`claude-agent-sdk` package) running against Amazon Bedrock. See the [root README](../README.md) for setup and AWS configuration.

## How it connects to Bedrock

The Claude Agent SDK works differently from the OpenAI Agents SDK:

- The package bundles the Claude Code CLI. `query(prompt=..., options=ClaudeAgentOptions(...))` runs it in the background and yields `AssistantMessage` / `ResultMessage` objects, so there's nothing else to install and no client object to register.
- Passing `CLAUDE_CODE_USE_BEDROCK=1` (plus `AWS_PROFILE` / `AWS_REGION`) through `options.env` makes the CLI use your AWS credentials instead of an Anthropic API key.
- The model is a Bedrock inference profile ID, read from `BEDROCK_CLAUDE_MODEL_ID` (defaults to `global.anthropic.claude-sonnet-5`).
- `tools=[]` disables Claude Code's built-in tools (file access, shell, and so on). If you enable them, the agent can act on the machine it runs on.
- `setting_sources=[]` stops the CLI from loading your `~/.claude` settings and the repo's `CLAUDE.md`. Without it, the agent picks up that context (for example, it mentions "the SDKs this repo uses").

Run all commands below from the repo root.

## Examples

| Script | Shows |
| --- | --- |
| `claude-agent.py` | A single question and a run summary |
| `claude-agent-streaming.py` | Streaming the response as it is generated |
| `claude-agent-tools.py` | Custom Python tools, served by an in-process MCP server |
| `claude-agent-conversation.py` | Conversation history with `ClaudeSDKClient` |
| `claude-agent-session.py` | Saved sessions: a fixed `session_id`, then `resume` from a new client |
| `claude-agent-workflow.py` | Multi-agent workflow orchestrated by code |
| `claude-agent-agents-as-tools.py` | Orchestration by an LLM, with subagents as tools |
| `claude-agent-handoffs.py` | Handing the conversation to another agent by resuming its session |
| `claude-agent-structured-output.py` | Getting a typed object back, with a schema in the prompt or in a tool |
| `claude-agent-guardrails.py` | Hooks that block a bad request and send a bad plan back for revision |

### Basic agent

`claude-agent.py` asks a Claude model on Bedrock a single question and prints the answer plus a run summary (turns, duration, cost):

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent.py
```

### Streaming

`claude-agent-streaming.py` prints the answer as it is generated, like `openai-agent-streaming.py` in the OpenAI folder. With `include_partial_messages=True`, `query()` also yields `StreamEvent` messages that carry the raw model stream events; the script prints the text from each `content_block_delta` event:

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-streaming.py
```

### Tools

`claude-agent-tools.py` is the Claude version of `openai-agent-tools.py`: a "Finance Assistant" compares buying a car with a loan against saving up for it, using a loan payment calculator and a savings goal calculator instead of doing the math itself. It prints each tool call and result before the answer.

Custom tools work differently from the OpenAI Agents SDK's `@function_tool`:

- `@tool(name, description, input_schema)` defines a tool. Each parameter is described with `Annotated[type, "description"]`, and the tool returns MCP-style content (`{"content": [{"type": "text", "text": ...}]}`).
- The agent doesn't take the tools directly. `create_sdk_mcp_server(...)` puts them in an MCP server that runs inside the Python process, and `mcp_servers={...}` gives it to the agent, which sees each tool as `mcp__<server>__<tool>`.
- Tools need permission to run, and a script has no one to ask, so `allowed_tools` pre-approves them. `tools=[]` still turns off the built-in tools only.
- Tool calls come back as `ToolUseBlock`s in `AssistantMessage`s, and their results as `ToolResultBlock`s in `UserMessage`s, matched by `tool_use_id`.

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-tools.py
```

### Conversation history

`claude-agent-conversation.py` is the Claude version of `openai-agent-conversation.py`. You tell a travel planner about a relaxed, nature-focused trip to Tokyo, then ask "What should I do on my first morning?" twice, with the same `ask(client, prompt)` helper. On `client_no_conversation`, which has only seen that question, the planner has to ask where you're going; on `client_with_conversation`, which was also asked turn 1, it gives a tailored suggestion. The script prints the session ID of each turn, so you can see which ones share a session.

With the OpenAI Agents SDK you keep the history yourself and pass it back with `to_input_list()`. Here you pass neither history nor a session ID:

- Each `ClaudeSDKClient` runs its own Claude Code CLI process, and that process remembers everything said on it. The client *is* the session: ask the same client again to continue, or use a new client to start fresh. (A one-off `query()` call also starts fresh.)
- Each turn is `client.query(...)` followed by reading the reply from `client.receive_response()`.
- The CLI saves each session as a transcript under `~/.claude/projects/`, and `get_session_messages(session_id)` reads it back, which the script uses to print both sessions' history. At the end, `delete_session(session_id)` removes both transcripts.

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-conversation.py
```

### Sessions

`claude-agent-session.py` is the Claude version of `openai-agent-session.py`. It has two turns with a travel planner, closes the client, then continues the conversation from a brand-new client, as if the app had restarted.

In `claude-agent-conversation.py` the conversation lived only as long as the client was open. Here a session ID brings it back:

- `ClaudeAgentOptions(session_id=...)` starts a session under an ID you choose. It must be a UUID, so the script derives a fixed one from the name `tokyo-trip` with `uuid.uuid5`.
- `ClaudeAgentOptions(resume=...)` makes a new client load that session's history before its first turn.
- There is no database to set up: the CLI already saves each session as `~/.claude/projects/<project folder>/<session id>.jsonl`, and `resume` reads it back. The script prints that path and the stored messages (`get_session_messages`).
- `delete_session(...)` removes the saved transcript, so the session can no longer be resumed. The script calls it at the end to clean up, and also at the start in case an earlier run stopped before its clean-up, so every run starts the same way, like `clear_session()` in the OpenAI version.

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-session.py
```

### Workflow orchestrated by code

`claude-agent-workflow.py` is the Claude version of `openai-agent-workflow.py`. Three planner agents (nature, culture, slow travel) draft a Tokyo day plan in parallel with `asyncio.gather`. An editor agent picks the best one, and a publisher agent saves it to `itineraries/claude/` (git-ignored) using a `save_itinerary` tool.

The Claude Agent SDK has no `Agent` class or `Runner`, so the script builds them from what the earlier examples used:

- An agent is a small dataclass (name, system prompt, tools), and `make_options(...)` turns it into `ClaudeAgentOptions`. Only the publisher gets an MCP server and `allowed_tools`.
- `run_agent(...)` runs one agent with its own `query()` call, so each agent is a separate Claude Code CLI process. The three planners' processes run at the same time.
- Each agent starts with no memory of the others. The code passes results along as prompts: the drafts go to the editor, and the editor's pick goes to the publisher.
- Each `query()` reports its own cost, so the script adds them up for the whole workflow.
- Each `query()` also saves its session as a transcript. The workflow never resumes them, so at the end the script deletes all five with `delete_session(...)`.

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-workflow.py
```

### Agents as tools (subagents)

`claude-agent-agents-as-tools.py` is the Claude version of `openai-agent-agents-as-tools.py`. An LLM does the orchestration instead of your code. A "Trip Director" agent calls the same three planners, compares their drafts and saves the winner with `save_itinerary`. The script prints each call it made.

The OpenAI Agents SDK wraps each agent with `agent.as_tool(...)`. The Claude Agent SDK uses subagents instead:

- `agents={name: AgentDefinition(description=..., prompt=...)}` defines the subagents. The `description` tells the director when to use one, and the `prompt` is the subagent's own system prompt. `tools=[]` gives the planners no tools, and `model="inherit"` runs them on the director's model.
- The director calls a subagent through Claude Code's built-in `Agent` tool, naming it in `subagent_type`. So `tools=["Agent"]` enables that one built-in tool, and `allowed_tools` pre-approves it along with `save_itinerary`.
- Each subagent starts with a fresh context and sees only the prompt the director writes for it. The description asks the director to pass only the traveler's request. Without that, the director added its own format instructions and the planners ignored their one-line format.
- Subagents run in the background by default, and the director only sometimes waits for them. In one test run it moved on without the drafts. `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` in `options.env` makes every `Agent` call wait and return the subagent's report as its result.
- Messages from inside a subagent carry a `parent_tool_use_id`, so the script skips them and prints only the director's own calls. The CLI wraps each report in a notice and a footer meant for the model, which the script strips.
- `ResultMessage.total_cost_usd` includes the subagents. `delete_session(...)` also removes their transcripts.

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-agents-as-tools.py
```

### Handoffs

`claude-agent-handoffs.py` is the Claude version of `openai-agent-handoffs.py`. The Trip Director still drafts with its three planner subagents, but it no longer saves the winner: it hands the conversation to a Publisher agent, which saves the plan and writes the final reply. The script labels each step with the agent in control.

A subagent reports back and the caller stays in charge. A handoff is the other thing: the next agent takes the conversation over. The OpenAI Agents SDK has `handoffs=[publisher]` for this; the Claude Agent SDK has no handoff, so the example builds one out of sessions:

- `transfer_to_publisher` is an MCP tool that does no work. Calling it *is* the handoff request, and it ends the director's run. Its one argument is a note on which planner won and why, which the script prints.
- A session is the conversation, so the code then runs the publisher with `resume=<the director's session id>` (see the sessions example). The publisher inherits the drafts and the director's choice, which is why it can summarize the winning plan without being told it.
- Resuming takes the *history*, not the agent: the publisher run passes its own `system_prompt`, its own `tools` and no subagents. Both runs append to the same session, so there is one transcript to delete at the end.
- `query()` always needs a prompt, so the handoff message is the prompt. Everything the publisher needs is in its instructions and the conversation it just inherited.
- The director's final answer is not the run's answer. The publisher's reply is, exactly as with a handoff in the OpenAI SDK.

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-handoffs.py
```

### Structured output

`claude-agent-structured-output.py` is the Claude version of `openai-agent-structured-output.py`. An "Itinerary Reviewer" reviews a deliberately overpacked, crowded Tokyo day plan and answers with an `ItineraryReview` object, so the script can read `review.crowd_risk` and `review.nature_score` in plain code to decide whether to approve the plan.

`ClaudeAgentOptions` has no `output_type`, so the script shows the two ways to get there. Both are built from the same Pydantic model:

- **Ask for JSON.** Put `ItineraryReview.model_json_schema()` in the system prompt, then parse the reply with `model_validate_json`. Nothing enforces the format: in a test run the model wrapped its JSON in a code fence after being told not to, so the parser strips a fence if it finds one. A reply that doesn't parse is your problem to handle.
- **Use a tool as the schema.** `@tool` accepts a full JSON Schema, so `@tool("submit_review", ..., ItineraryReview.model_json_schema())` turns the model into a form the agent has to fill in. The review arrives as the tool call's arguments, which the script reads off the `ToolUseBlock` and passes to `model_validate`.

The second way is the sturdier one. `create_sdk_mcp_server` checks each call against the schema with `jsonschema` before your handler runs, and a mismatch goes back to the agent as an error it can correct, so malformed output never reaches your code. The script takes the last `submit_review` call for that reason.

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-structured-output.py

# have a planner agent write a fresh itinerary to review instead of the built-in bad sample
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-structured-output.py --planner chaotic     # overpacked and crowded
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-structured-output.py --planner thoughtful  # relaxed and nature-focused
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-structured-output.py --planner offbeat     # lesser-known spots off the beaten path
```

### Guardrails

`claude-agent-guardrails.py` is the Claude version of `openai-agent-guardrails.py`. Three planners (thoughtful, offbeat, chaotic) answer the same request behind the same two guardrails: one that rejects requests that aren't about Tokyo travel, and one that refuses to deliver a plan that is unrealistic, crowded, or barely about nature.

The OpenAI Agents SDK has `input_guardrails` and `output_guardrails`, and a tripwire raises an exception. The Claude Agent SDK has **hooks**: callbacks the CLI runs at fixed points in the agent loop, which can block what happens next. Each guardrail here is one hook, and each hook runs a checker agent that answers through a tool schema, as in the structured output example.

- **Input guardrail — a `UserPromptSubmit` hook.** It runs a Topic Checker on the prompt and returns `{"decision": "block", "reason": ...}` for anything off topic. The prompt is then thrown away: the planner run comes back with `num_turns` of 0, so an off-topic request costs one small check instead of a plan.
- **Output guardrail — a `PreToolUse` hook.** Hooks see tool calls, not the final reply, so the planner delivers by calling `submit_plan` and `HookMatcher(matcher="mcp__guarded__submit_plan", ...)` guards that call. The hook runs an Itinerary Reviewer on the plan and answers with `permissionDecision` of `"allow"` or `"deny"`.
- **A denial is feedback, not just a stop.** `permissionDecisionReason` goes back to the planner as the tool's result, so it can fix the plan and submit again — something an OpenAI tripwire can't do. In a test run the Thoughtful Planner's first plan was denied for putting Shinjuku Gyoen and Yoyogi Park at cherry blossom time; it came back with the National Institute for Nature Study and Todoroki Valley, which passed.
- **The second failure ends it.** The guardrail then returns `{"continue_": False, "stopReason": ...}`, which stops the whole run. The Chaotic Planner reliably gets that far, and nothing is delivered. Note that the run still reports `is_error=False` with empty text, so the script tracks the block in the guardrail object rather than reading it off `ResultMessage`.
- Both guardrails are plain dataclasses whose `hook` method is the callback, so each one keeps the decisions it made for the script to print afterwards. Hook timeouts default to 60s, and these hooks run a whole agent, so `HookMatcher(timeout=180)` gives them room.

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-guardrails.py

# one planner at a time
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-guardrails.py --planner thoughtful  # passes, usually after one revision
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-guardrails.py --planner chaotic     # denied twice, run stopped

# trip the input guardrail instead
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-guardrails.py --planner thoughtful --request "Write me a Python script to rename files in a folder."
```
