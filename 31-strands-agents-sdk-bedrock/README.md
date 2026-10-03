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
| `strands-agent-tools.py` | Custom Python tools, defined with `@tool` |
| `strands-agent-conversation.py` | Conversation history kept by the `Agent` object |
| `strands-agent-session.py` | Saved sessions: `FileSessionManager` writes the history to disk, and a new agent reloads it |
| `strands-agent-workflow.py` | Multi-agent workflow orchestrated by code |
| `strands-agent-agents-as-tools.py` | Orchestration by an LLM, with agents as tools (`agent.as_tool()`) |

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

### Tools

`strands-agent-tools.py` is the Strands version of `openai-agent-tools.py` and `claude-agent-tools.py`. A "Finance Assistant" compares buying a car with a loan against saving up for it. It uses a loan payment calculator and a savings goal calculator instead of doing the math itself. The script prints each tool call and its result before the answer.

Of the three SDKs, Strands needs the least code for tools:

- `@tool` reads everything from the function. The name comes from the function name, and the types come from the type hints. The docstring's first line becomes the description, and its `Args:` section describes each parameter. The script prints `tool_spec`, the JSON schema that Strands sends to the model.
- A tool returns a plain value, here a string. It doesn't need MCP-style content blocks as in the Claude Agent SDK. The decorated function also stays callable as ordinary Python.
- `Agent(tools=[...])` takes the functions directly. There is no MCP server and no `allowed_tools` list, and Strands runs a tool in this process whenever the model asks for one.
- To see the calls, read `agent.messages`, the conversation in Converse format. A call is a `toolUse` block in an assistant message, and its result is a `toolResult` block in the next user message, matched by `toolUseId`.
- `Cycles` in the run summary counts model calls. A round of tool calls adds one. In test runs, Sonnet and Nova asked for both tools at once (2 cycles), and gpt-oss called them one at a time (3 cycles).

The tools fix the arithmetic, not the reasoning. All three models got the same numbers from the tools, but their conclusions varied. In one test, gpt-oss said saving meant waiting "~7 months longer", and Nova said the loan costs less overall.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-tools.py
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-tools.py
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-tools.py
```

### Conversation history

`strands-agent-conversation.py` is the Strands version of `openai-agent-conversation.py` and `claude-agent-conversation.py`. You tell a travel planner about a relaxed, nature-focused trip to Tokyo. Then you ask "What should I do on my first morning?" twice, with the same `ask(agent, prompt)` helper:

- `agent_no_conversation` has seen only that question, so the planner asks where you're going.
- `agent_with_conversation` was also asked turn 1, so it suggests a quiet Tokyo garden.

The three SDKs keep history in three different places. In OpenAI you pass it back yourself with `to_input_list()`, and in Claude the CLI process holds it. In Strands it's the **`Agent` object**:

- Each call appends the prompt and the reply to `agent.messages`, and the next call sends that whole list to the model. To continue a conversation, call the same agent again. To start fresh, make another agent. The script prints how many messages each agent holds before each turn.
- `agent.messages` is a plain Python list in Converse format (`{"role": ..., "content": [blocks]}`), so you can read it, save it or edit it. There's no transcript file and nothing to delete. The history lives only as long as the object does (the sessions example will persist it).
- Both agents share one `BedrockModel`. The model holds only the connection settings, not the conversation, so sharing it doesn't mix the histories.
- gpt-oss's reasoning is stored in the history too, as a `reasoningContent` block next to the text. `print_history` shows only text blocks.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-conversation.py
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-conversation.py
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-conversation.py
```

### Sessions

`strands-agent-session.py` is the Strands version of `openai-agent-session.py` and `claude-agent-session.py`. It has two turns with a travel planner, then drops the agent and builds a brand-new one from the same session ID, as if the app had restarted. The new agent continues the conversation.

In the conversation example, the history lived only in `agent.messages` and was lost with the object. A **session manager** also saves it:

- `Agent(session_manager=FileSessionManager(session_id=..., storage_dir=...), agent_id=...)` attaches the agent to a session. Each new message is written to disk as it is added, so there's no save call.
- When an agent is created with a session that already holds messages for its `agent_id`, they are loaded into `agent.messages` at once, before the first call. The script prints the reloaded history before turn 3 to show this.
- The store is a folder of plain JSON files: `session.json`, `agent.json`, and one `message_<n>.json` per message. The script lists them. It uses `sessions/` next to the script (git-ignored); without `storage_dir` they go to `~/.strands/sessions/`.
- A session can hold several agents, each under its own `agent_id`, which is why the files are nested under `agents/agent_travel-planner/`.
- The system prompt is not saved. It comes from the code each time, as in the OpenAI and Claude versions. `agent.json` does save the agent's `state` and its conversation manager's settings.
- Creating the manager creates the session, so `delete_session(...)` always has something to remove. The script calls it at the start, so every run starts the same way, and at the end to clean up.
- `S3SessionManager` has the same interface and keeps the files in an S3 bucket instead, for apps that run on more than one machine.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-session.py
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-session.py
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-session.py
```

### Workflow orchestrated by code

`strands-agent-workflow.py` is the Strands version of `openai-agent-workflow.py` and `claude-agent-workflow.py`. Three planner agents (nature, culture, slow travel) draft a Tokyo day plan in parallel. An editor agent picks the best one, and a publisher agent saves it to `itineraries/strands/` (git-ignored) using a `save_itinerary` tool.

It is close to the OpenAI version, because Strands has a real `Agent` class:

- Each agent is an `Agent(name=..., model=..., system_prompt=..., tools=...)`, built by a small `make_agent` factory. All five share one `BedrockModel`, and only the publisher has a tool.
- `await agent.invoke_async(prompt)` is the awaitable form of `agent(prompt)`, so `asyncio.gather` runs the three planners at the same time. There's no subprocess per agent, as there is in the Claude version.
- Each agent keeps its own `agent.messages`, so the agents never see each other's conversations. The code passes results along as prompts: the drafts go to the editor, and the editor's pick goes to the publisher.
- The publisher's tool call is read from `publisher.messages`, as in the tools example.
- Every agent tracks its own token usage in `agent.event_loop_metrics`, and the script adds them up. Nothing is saved besides the itinerary, so there's nothing to clean up.

Strands also has multi-agent classes in `strands.multiagent`: `GraphBuilder` for a fixed flow of agents and `Swarm` for agents that hand work to each other. This example keeps the flow in plain Python, to match the other two folders.

In test runs, the whole workflow took 40.5s on Sonnet, 22.9s on gpt-oss and 7.3s on Nova. Nova drafted all three plans in 2.8s.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-workflow.py
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-workflow.py
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-workflow.py
```

### Agents as tools

`strands-agent-agents-as-tools.py` is the Strands version of `openai-agent-agents-as-tools.py` and `claude-agent-agents-as-tools.py`. An LLM does the orchestration instead of your code. A "Trip Director" agent calls the same three planners, compares their drafts and saves the winner with `save_itinerary`. The script prints each call it made.

Strands works like the OpenAI Agents SDK here. The Claude Agent SDK needed subagents and a background-task setting instead:

- `planner.as_tool(name=..., description=...)` wraps an agent as a tool with one string parameter, `input`. Calling the tool runs that agent on the input and returns its reply, and then control goes back to the director. Tool names can't contain spaces, so the script derives `nature_guide` from "Nature Guide". You can also put an `Agent` straight into `tools=[...]`, and Strands calls `as_tool()` for you with the agent's own name.
- Each call starts the planner from the conversation it had when it was wrapped (`preserve_context=False`, the default), so calling the same planner twice doesn't mix the drafts.
- The director's tools mix both kinds, and the agent graph shows each one's `tool_type`: `agent` for the planners, `function` for `save_itinerary`.
- Calls to the planners are ordinary `toolUse` / `toolResult` blocks in `director.messages`, so the steps are read as in the tools example. The planners' own model calls stay in their own agents.
- When the model asks for several tools at once, Strands runs them concurrently: the default `tool_executor` is `ConcurrentToolExecutor`. In test runs, Sonnet and Nova called all three planners in one cycle (3 cycles in total), while gpt-oss called them one at a time (5 cycles).
- Each agent counts only its own tokens, so the run summary reports the director and the planners separately.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-agents-as-tools.py
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-agents-as-tools.py
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-agents-as-tools.py
```
