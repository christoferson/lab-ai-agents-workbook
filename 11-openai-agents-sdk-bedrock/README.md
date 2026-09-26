# OpenAI Agents SDK on Amazon Bedrock

Examples of the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) (`agents` package) running against Amazon Bedrock instead of the OpenAI API. See the [root README](../README.md) for setup and AWS configuration.

## How it connects to Bedrock

Every script starts with the same setup:

- `AsyncOpenAI(provider=bedrock(region=...))` from `openai.providers` creates a client that talks to Bedrock.
- `set_default_openai_client(...)` registers it, so every `Agent(...)` / `Runner.run(...)` routes to Bedrock.
- `set_tracing_disabled(True)` turns off tracing, since there is no OpenAI tracing endpoint to export to.
- The model is read from `BEDROCK_MODEL_ID` (defaults to `openai.gpt-5.6-luna`).

Run all commands below from the repo root.

## Examples

| Script | Shows |
| --- | --- |
| `openai-agent.py` | A basic agent and the full run result |
| `openai-agent-streaming.py` | Streaming the response token by token |
| `openai-agent-tools.py` | Function tools |
| `openai-agent-conversation.py` | Conversation history with `to_input_list()` |
| `openai-agent-session.py` | Automatic history with `SQLiteSession` |
| `openai-agent-workflow.py` | Multi-agent workflow orchestrated by code |
| `openai-agent-agents-as-tools.py` | Orchestration by an LLM, with agents as tools |
| `openai-agent-handoffs.py` | Handoffs between agents |
| `openai-agent-structured-output.py` | Structured output with a Pydantic model |
| `openai-agent-guardrails.py` | Input and output guardrails |
| `openai-agent-deep-research.py` | A deep research pipeline: plan, search, write, fact-check, publish |

### Basic agent

`openai-agent.py` prints the active AWS profile and region, then sends a prompt to the agent and prints the response, followed by the full run result (items, raw responses, usage) pretty-printed with [rich](https://github.com/Textualize/rich):

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent.py
```

### Streaming

`openai-agent-streaming.py` streams the response token by token instead:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-streaming.py
```

### Tools

In `openai-agent-tools.py`, a "Finance Assistant" agent compares buying a car with a loan against saving up for it. It calls local Python tools (a loan payment calculator and a savings goal calculator) instead of doing the math itself, and prints each tool call and result before its answer:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-tools.py
```

### Conversation history

In `openai-agent-conversation.py`, you tell a travel planner about a relaxed, nature-focused trip to Tokyo, then ask "What should I do on my first morning?" twice. The first time is a fresh run, so the planner has to ask where you're going. The second time the previous turn is passed back via `to_input_list()`, so it gives a tailored suggestion:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-conversation.py
```

### Sessions

`openai-agent-session.py` has the SDK manage history for you. It passes a `SQLiteSession` to `Runner.run(..., session=session)`, which loads and saves each turn automatically. The history is stored in `sessions.db` next to the script (git-ignored), and the demo closes and reopens the session to show the conversation survives a restart. No extra dependency is needed, since it uses Python's built-in `sqlite3`:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-session.py
```

### Workflow orchestrated by code

In `openai-agent-workflow.py`, three planner agents (nature, culture, slow travel) draft a Tokyo day plan in parallel with `asyncio.gather`. An editor agent picks the best one, and a publisher agent saves it to `itineraries/` (git-ignored) using a `save_itinerary` tool:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-workflow.py
```

### Agents as tools

`openai-agent-agents-as-tools.py` lets an LLM do the orchestration instead. The same three planners are wrapped with `agent.as_tool(...)` and given, together with `save_itinerary`, to a "Trip Director" planning agent. The director decides on its own to call each planner, compare the drafts and save the winner, and the script prints each call it made:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-agents-as-tools.py
```

### Handoffs

In `openai-agent-handoffs.py`, the Trip Director still uses the planners as tools, but instead of saving the plan itself it hands off to a Publisher agent (`handoffs=[publisher]`). A tool call returns control to the caller; a handoff transfers the conversation, so the Publisher saves the plan and writes the final reply. The script labels each step with the agent in control:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-handoffs.py
```

### Structured output

In `openai-agent-structured-output.py`, an "Itinerary Reviewer" agent with `output_type=ItineraryReview` (a Pydantic model) reviews a deliberately overpacked, crowded Tokyo day plan. `result.final_output` is an `ItineraryReview` object rather than text, so the script reads fields like `review.crowd_risk` and `review.nature_score` directly to decide whether to approve the plan:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-structured-output.py

# have a planner agent write a fresh itinerary to review instead of the built-in bad sample
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-structured-output.py --planner chaotic     # overpacked and crowded
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-structured-output.py --planner thoughtful  # relaxed and nature-focused
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-structured-output.py --planner offbeat     # lesser-known spots off the beaten path
```

### Guardrails

In `openai-agent-guardrails.py`, each planner has two guardrails:

- An input guardrail runs a Topic Checker on the request and rejects anything that isn't about Tokyo travel. It uses `run_in_parallel=False`, so it finishes before the planner starts and a rejected request spends no planner tokens.
- An output guardrail runs an Itinerary Reviewer (structured output) on each plan and trips if the plan is unrealistic, crowded, or not nature-focused.

By default the same request goes to every planner. The Thoughtful and Offbeat Planners' plans should pass, and the Chaotic Planner's plan should be blocked with an `OutputGuardrailTripwireTriggered` exception. The script shows the review behind each decision:

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-guardrails.py

# run just one planner: thoughtful, offbeat or chaotic
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-guardrails.py --planner thoughtful

# send an off-topic request to trip the input guardrail
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-guardrails.py --planner thoughtful --request "Write my Python homework for me."
```

### Deep research

`openai-agent-deep-research.py` combines the earlier patterns into a research pipeline, with code deciding the order of five agents. Models don't know today's date, so every agent's instructions include it (in Japan time). This is cheaper and more reliable than a "current time" tool, which would cost an extra model round trip and which the agent might forget to call.

1. A Planner (structured output `SearchPlan`) turns the query into 3 searches, each with a reason.
2. A Searcher runs every search in parallel with `asyncio.gather`, using `WebSearchTool`. `tool_choice="required"` forces a search, so each summary is grounded in current results rather than the model's memory. The `url_citation` sources from each answer are passed along.
3. A Writer (structured output `ReportData`) combines the summaries into a Markdown report with a Sources section, a short summary and follow-up questions.
4. A Fact Checker (structured output `FactCheck`) compares the report against the summaries and can run its own web searches to verify doubtful claims. It flags contradictions, unsupported claims and outdated information. If it finds issues, the Writer revises the report once to fix them.
5. A Publisher saves the report to `reports/` (git-ignored) with a `save_report` tool.

[Web Search](https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html) is a Bedrock-hosted tool. Its requirements:

- It runs on the `bedrock-mantle` endpoint with the Responses API, which is what the `bedrock()` provider and the Agents SDK use by default.
- It needs a supported model, such as the `openai.gpt-5.6` family, and region (`us-east-1`, `us-east-2` or `us-west-2`).
- Your IAM identity needs `bedrock-websearch:InvokeSearch` and `bedrock-websearch:InvokeFetch`, which `AmazonBedrockFullAccess` grants.

The script sets `external_web_access=False`, so searches are served from the Bedrock web index and cache and your request data stays inside AWS. Setting it to `True` also needs the `bedrock-websearch:ExternalWebAccess` permission.

The SDK's `WebSearchTool` always sends `filters` and `user_location` fields, even when they're empty, and Bedrock rejects them with a 400 error. The Searcher therefore passes a clean tool definition through `ModelSettings(extra_body={"tools": [...]})`, which replaces the SDK's `tools` list in the request.

```bash
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-deep-research.py

# research your own question
uv run --env-file .env 11-openai-agents-sdk-bedrock/openai-agent-deep-research.py --query "A quiet day trip from Tokyo with hiking and a hot spring"
```
