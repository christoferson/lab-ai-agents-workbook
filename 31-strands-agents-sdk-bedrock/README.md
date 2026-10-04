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
| `strands-agent-handoffs.py` | Handing the task to another agent with a `Swarm` |
| `strands-agent-structured-output.py` | A typed Pydantic object as the answer (`structured_output_model`) |
| `strands-agent-guardrails.py` | Input and output guardrails, as interventions |

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

### Handoffs

`strands-agent-handoffs.py` is the Strands version of `openai-agent-handoffs.py` and `claude-agent-handoffs.py`. The Trip Director still drafts with its three planner tools, but it no longer saves the winner. Instead it hands the task to a Publisher agent, which saves the plan and writes the final reply. The script labels each step with the agent in control.

A tool reports back and the caller stays in charge. A handoff is the other thing: the next agent takes over. In Strands that is a **`Swarm`**, a group of agents that pass work to each other:

- `Swarm([director, publisher], entry_point=director)` gives every agent in it a `handoff_to_agent(agent_name, message, context)` tool. Calling it ends that agent's turn, and the swarm starts the named agent next. An agent that finishes without handing off ends the swarm.
- The agents' names are the swarm's node IDs, and the director passes one as `agent_name`, so they are written like identifiers (`trip_director`, `publisher`). The publisher's `description` is how the other agents learn what it does.
- **The next agent doesn't inherit the conversation.** This is the big difference from the other two SDKs. An OpenAI handoff passes the whole history, and the Claude example resumes the session. Swarm instead builds the next agent a new prompt: the handoff `message`, the original task, which agents worked on it, any `context` that was passed, and the other agents it could hand off to. The script prints that prompt in full.
- So the publisher never sees the three drafts. The director's instructions tell it to put the full chosen plan in the handoff message, and in test runs all three models did (gpt-oss sometimes put it in `context` instead). The OpenAI and Claude examples don't need that instruction.
- The steps are in the director's system prompt, not the task, because Swarm repeats the task in every agent's prompt. When the task held the steps, Nova's publisher read them as its own job and handed back to the director.
- `result.node_history` lists the agents in the order they had control. `max_handoffs` counts agent turns, not handoffs, and reaching it ends the swarm with status `failed`.
- Before each of its turns, an agent's `messages` are reset to how they were at the start. So an agent that had control twice shows only its last turn, and the script says so.
- That reset doesn't clear the agent's token counter. Swarm's `result.accumulated_usage` adds up each agent's running total once per turn, so it overcounts an agent that had control twice. The summary uses each agent's own `event_loop_metrics` instead.

`as_tool(delegate=True)` is a lighter alternative: the sub-agent's reply becomes the caller's final answer, with no extra model call. But it is still a tool call, so the sub-agent gets only its `input` string and the caller's run ends with it.

In test runs, Sonnet took about 36s and Nova 11s. gpt-oss varied from 15s to 160s between identical runs, and the debug logs showed the time was spent waiting for the model's streamed replies.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-handoffs.py
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-handoffs.py
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-handoffs.py
```

### Structured output

`strands-agent-structured-output.py` is the Strands version of `openai-agent-structured-output.py` and `claude-agent-structured-output.py`. An "Itinerary Reviewer" grades a deliberately bad Tokyo plan and answers with an `ItineraryReview` Pydantic object instead of text. Plain code then reads its fields to approve the plan or send it back. `--planner chaotic|thoughtful|offbeat` has a planner agent write a fresh plan to review instead.

Strands has this built in, like `output_type` in the OpenAI Agents SDK. The Claude Agent SDK has nothing equivalent, so its example builds two ways by hand:

- `Agent(structured_output_model=ItineraryReview)` sets it for every call, and `agent(prompt, structured_output_model=...)` sets it for one call. `result.structured_output` is the parsed object, and `str(result)` isn't needed.
- Underneath it is the Claude example's second way, done for you. Strands adds a tool named after the class (`ItineraryReview`), with the class's JSON schema as its input, and the model answers by calling it. The script prints that call and its result from `agent.messages`.
- The tool validates the arguments with Pydantic. If they don't fit, the errors go back to the model as the tool result, so it can fix them and call again.
- According to the Strands source (`event_loop.py`), if the model ends its turn without calling the tool, Strands sends a follow-up prompt and forces the tool with `toolChoice`. In test runs no model needed this. All three called the tool on the first cycle, so `Stop reason` is `tool_use` and `Cycles` is 1.
- Field descriptions go to the model as part of the schema, so they double as instructions, as in the other two folders.

In test runs, the review of the sample plan took 13.0s on Sonnet, 8.2s on gpt-oss and 3.0s on Nova. All three marked it unrealistic with high crowd risk.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-structured-output.py
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-structured-output.py --planner thoughtful
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-structured-output.py
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-structured-output.py
```

### Guardrails

`strands-agent-guardrails.py` is the Strands version of `openai-agent-guardrails.py` and `claude-agent-guardrails.py`. Three planners (thoughtful, offbeat, chaotic) answer the same request behind the same two guardrails: one rejects requests that aren't about Tokyo travel, and one refuses to deliver a plan that is unrealistic, crowded or barely about nature. Each guardrail runs a checker agent that answers with `structured_output_model`, as in the structured output example.

The OpenAI Agents SDK has guardrails that raise an exception, and the Claude Agent SDK has hooks. Strands has **interventions**: `Agent(interventions=[...])` takes `InterventionHandler` subclasses, and Strands calls the lifecycle methods each one overrides (`before_invocation`, `before_model_call`, `after_model_call`, `before_tool_call`, `after_tool_call`). Each method returns an action: `Proceed`, `Deny`, `Guide`, `Confirm` or `Transform`. The handlers run in list order, so the cheap topic check goes first.

- **Input guardrail: `before_invocation` returns `Deny`.** It runs a Topic Checker on the request. `Deny(reason=...)` cancels the call before the planner's model runs, so in a test the off-topic request cost the planner 0 tokens. The reply is `"DENIED: <reason>"` with stop reason `end_turn`, and nothing on the result says it was denied, so the guardrail object keeps its decisions for the script to read.
- **After a `Deny`, don't reuse the agent.** Its `messages` hold only the `DENIED` reply, not the request, and the next call fails because Bedrock wants a conversation to start with a user message. The script builds a fresh planner for each run.
- **Output guardrail: `after_model_call` returns `Guide`.** This runs after every model call, so it reviews only finished replies (`stop_reason == "end_turn"`). It works on the final reply itself, with no tool to route the plan through as in the Claude version.
- **`Guide(feedback=...)` throws the reply away and calls the model again**, with the feedback added as a user message (`[itinerary_quality] Revise the plan...`). The rejected plan isn't kept in the history. Strands has no limit on these retries, so the guardrail counts them and asks for one revision at most.
- **The output guardrail can't block.** `Deny` does nothing after a model call (Strands logs a warning), so the second failed plan still comes back as the result. The script withholds it, like the Claude version, which also tracks the block itself.

In test runs, feedback often fixed the plan. All three models' chaotic planners turned a plan with nature 1 or 2 into one that passed, which an OpenAI tripwire can't do and the Claude version's chaotic planner reliably failed to do. gpt-oss's thoughtful and offbeat planners failed the realism check twice and were withheld. Nova's chaotic planner sometimes ignored its instructions and wrote a good plan first time. In one Nova run, an agent stopped at the model's output token limit and the script exited with `MaxTokensReachedException`; two reruns didn't repeat it.

```bash
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-guardrails.py
STRANDS_MODEL_PROVIDER=openai uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-guardrails.py
STRANDS_MODEL_PROVIDER=amazon uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-guardrails.py

# one planner at a time
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-guardrails.py --planner chaotic

# trip the input guardrail instead
uv run --env-file .env 31-strands-agents-sdk-bedrock/strands-agent-guardrails.py --planner thoughtful --request "Write me a Python script to rename files in a folder."
```
