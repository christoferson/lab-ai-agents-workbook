# Strands Agents SDK on Amazon Bedrock

Examples of the [Strands Agents SDK](https://strandsagents.com/) (`strands-agents` package) running against Amazon Bedrock. See the [root README](../README.md) for setup and AWS configuration.

## How it connects to Bedrock

Strands is AWS's own agent SDK, and Bedrock is its default provider:

- An `Agent` is a model plus a system prompt. Calling the agent, `agent(prompt)`, runs its loop and returns an `AgentResult`. The call is synchronous, so there's no `asyncio` in the basic example.
- Strands calls Bedrock directly. It needs no subprocess (as in the Claude Agent SDK) and no client to register (as in the OpenAI Agents SDK). The script builds a boto3 session from `AWS_PROFILE` / `AWS_REGION` and gives it to the model.
- A bare `Agent()` with no model also uses Bedrock, but with a default model and whatever AWS credentials boto3 finds, so the examples always pass `model=` explicitly.

## Choosing the model: Claude, gpt-oss or Nova

`STRANDS_MODEL_PROVIDER` picks the model family, and a separate variable holds each family's model ID:

| `STRANDS_MODEL_PROVIDER` | Model ID variable (default) |
| --- | --- |
| `anthropic` (default) | `STRANDS_MODEL_ID_ANTHROPIC` (`global.anthropic.claude-sonnet-5`) |
| `openai` | `STRANDS_MODEL_ID_OPENAI` (`openai.gpt-oss-120b-1:0`) |
| `amazon` | `STRANDS_MODEL_ID_AMAZON` (`global.amazon.nova-2-lite-v1:0`) |

The `openai` option uses OpenAI's open-weight [gpt-oss models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-openai.html) (`openai.gpt-oss-120b-1:0` or `openai.gpt-oss-20b-1:0`). The `amazon` option uses Amazon's own [Nova 2 Lite](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html), called through its `global.` inference profile. In `us-east-1`, the plain `amazon.nova-2-lite-v1:0` isn't available In-Region.

All three run on Bedrock's Converse API, so the same `BedrockModel` class works for each, and only the model ID changes. gpt-oss also returns its reasoning, but `str(result)` includes only the answer text.

The OpenAI examples' `openai.gpt-5.6-luna` (`BEDROCK_MODEL_ID`) doesn't work here: the Converse API rejects it ("on-demand throughput isn't supported").

```bash
# Claude (default)
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent.py

# gpt-oss
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent.py

# Nova
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent.py
```

You can also set `STRANDS_MODEL_PROVIDER` in `.env`.
- By default, the agent prints the response as it streams in. `callback_handler=None` turns that off, so the script prints the result itself.

Run all commands below from the repo root.

## Examples

| Script | Shows |
| --- | --- |
| `strands-agent.py` | A single question and a run summary |
| `strands-agent-streaming.py` | Streaming the response as it is generated |

### Basic agent

`strands-agent.py` asks the chosen model a single question. It prints the answer and a run summary: stop reason, cycles, duration and token usage. Strands reports tokens rather than cost.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent.py
```

### Streaming

`strands-agent-streaming.py` prints the answer as it is generated, like the streaming examples in the OpenAI and Claude folders. It uses the same model and agent as the basic example. Only the call changes:

- `agent.stream_async(prompt)` is an async iterator of plain dicts instead of a single `AgentResult`. A text chunk has a `"data"` key, and the last event has a `"result"` key with the same `AgentResult` that `agent(prompt)` returns, so the run summary works as before.
- Nothing has to be turned on. `BedrockModel` always calls Bedrock's streaming API (ConverseStream); `agent(prompt)` just waits for the end. That is also why `callback_handler=None` matters here: the default handler prints every `"data"` chunk itself, so without it each chunk would print twice.
- Other keys carry the rest of the stream. `"event"` holds the raw Bedrock stream events, and gpt-oss's reasoning arrives separately as `"reasoningText"`, so printing only `"data"` shows just the answer.
- The run summary adds the time to the first chunk, which is what streaming improves: in a test the full answer took 6.0s on Sonnet, but the first text appeared after 3.7s.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-streaming.py
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-streaming.py
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-streaming.py
```
