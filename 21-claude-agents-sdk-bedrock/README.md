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
