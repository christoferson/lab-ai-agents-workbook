# Strands Agents SDK on Amazon Bedrock

Examples of the [Strands Agents SDK](https://strandsagents.com/) (`strands-agents` package) running against Amazon Bedrock. See the [root README](../README.md) for setup and AWS configuration.

## How it connects to Bedrock

Strands is AWS's own agent SDK, and Bedrock is its default provider:

- An `Agent` is a model plus a system prompt. Calling the agent, `agent(prompt)`, runs its loop and returns an `AgentResult`. The call is synchronous, so there's no `asyncio` in the basic example.
- `BedrockModel(model_id=..., boto_session=...)` calls Bedrock directly through boto3. It needs no subprocess (as in the Claude Agent SDK) and no client to register (as in the OpenAI Agents SDK). The script builds the boto3 session from `AWS_PROFILE` / `AWS_REGION`.
- A bare `Agent()` with no model also uses Bedrock, but with a default model and whatever AWS credentials boto3 finds, so the examples always pass `model=` explicitly.
- The model is a Bedrock inference profile ID, read from `BEDROCK_CLAUDE_MODEL_ID` (defaults to `global.anthropic.claude-sonnet-5`), the same variable the Claude Agent SDK examples use.
- By default, the agent prints the response as it streams in. `callback_handler=None` turns that off, so the script prints the result itself.

Run all commands below from the repo root.

## Examples

| Script | Shows |
| --- | --- |
| `strands-agent.py` | A single question and a run summary |

### Basic agent

`strands-agent.py` asks Claude on Bedrock a single question. It prints the answer and a run summary: stop reason, cycles, duration and token usage. Strands reports tokens rather than cost.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent.py
```
