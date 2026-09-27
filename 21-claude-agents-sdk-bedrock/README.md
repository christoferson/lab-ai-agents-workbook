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
