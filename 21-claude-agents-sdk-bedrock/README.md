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
- The CLI saves each session as a transcript under `~/.claude/projects/`, and `get_session_messages(session_id)` reads it back, which the script uses to print both sessions' history.

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent-conversation.py
```
