# Claude Agent SDK on Amazon Bedrock

Examples of the [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview) (`claude-agent-sdk` package) running against Amazon Bedrock. See the [root README](../README.md) for setup and AWS configuration.

## How it connects to Bedrock

The Claude Agent SDK works differently from the OpenAI Agents SDK:

- The package bundles the Claude Code CLI. `query(prompt=..., options=ClaudeAgentOptions(...))` runs it in the background and yields `AssistantMessage` / `ResultMessage` objects, so there's nothing else to install and no client object to register.
- Passing `CLAUDE_CODE_USE_BEDROCK=1` (plus `AWS_PROFILE` / `AWS_REGION`) through `options.env` makes the CLI use your AWS credentials instead of an Anthropic API key.
- The model is a Bedrock inference profile ID, read from `BEDROCK_CLAUDE_MODEL_ID` (defaults to `global.anthropic.claude-sonnet-5`).
- `tools=[]` disables Claude Code's built-in tools (file access, shell, and so on). If you enable them, the agent can act on the machine it runs on.

Run all commands below from the repo root.

## Examples

| Script | Shows |
| --- | --- |
| `claude-agent.py` | A single question and a run summary |

### Basic agent

`claude-agent.py` asks a Claude model on Bedrock a single question and prints the answer plus a run summary (turns, duration, cost):

```bash
uv run --env-file .env 21-claude-agents-sdk-bedrock/claude-agent.py
```
