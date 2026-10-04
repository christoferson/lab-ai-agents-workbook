# SDK comparison

The same examples in the OpenAI Agents SDK, the Claude Agent SDK and the Strands Agents SDK, all running on Amazon Bedrock. Each aspect of an example has a divider row that explains how the SDKs differ, followed by a row with only the code for it in each SDK. Follow the links for the full scripts.

| Example | OpenAI Agents SDK | Claude Agent SDK | Strands Agents SDK |
| --- | --- | --- | --- |
| [Basic agent](#1-basic-agent) | [`openai-agent.py`](11-openai-agents-sdk-bedrock/openai-agent.py) | [`claude-agent.py`](21-claude-agents-sdk-bedrock/claude-agent.py) | [`strands-agent.py`](31-strands-agents-sdk-bedrock/strands-agent.py) |
| [Streaming](#2-streaming) | [`openai-agent-streaming.py`](11-openai-agents-sdk-bedrock/openai-agent-streaming.py) | [`claude-agent-streaming.py`](21-claude-agents-sdk-bedrock/claude-agent-streaming.py) | [`strands-agent-streaming.py`](31-strands-agents-sdk-bedrock/strands-agent-streaming.py) |
| [Tools](#3-tools) | [`openai-agent-tools.py`](11-openai-agents-sdk-bedrock/openai-agent-tools.py) | [`claude-agent-tools.py`](21-claude-agents-sdk-bedrock/claude-agent-tools.py) | [`strands-agent-tools.py`](31-strands-agents-sdk-bedrock/strands-agent-tools.py) |
| [Conversation history](#4-conversation-history) | [`openai-agent-conversation.py`](11-openai-agents-sdk-bedrock/openai-agent-conversation.py) | [`claude-agent-conversation.py`](21-claude-agents-sdk-bedrock/claude-agent-conversation.py) | [`strands-agent-conversation.py`](31-strands-agents-sdk-bedrock/strands-agent-conversation.py) |
| [Sessions](#5-sessions) | [`openai-agent-session.py`](11-openai-agents-sdk-bedrock/openai-agent-session.py) | [`claude-agent-session.py`](21-claude-agents-sdk-bedrock/claude-agent-session.py) | [`strands-agent-session.py`](31-strands-agents-sdk-bedrock/strands-agent-session.py) |
| [Workflow orchestrated by code](#6-workflow-orchestrated-by-code) | [`openai-agent-workflow.py`](11-openai-agents-sdk-bedrock/openai-agent-workflow.py) | [`claude-agent-workflow.py`](21-claude-agents-sdk-bedrock/claude-agent-workflow.py) | [`strands-agent-workflow.py`](31-strands-agents-sdk-bedrock/strands-agent-workflow.py) |
| [Agents as tools](#7-agents-as-tools) | [`openai-agent-agents-as-tools.py`](11-openai-agents-sdk-bedrock/openai-agent-agents-as-tools.py) | [`claude-agent-agents-as-tools.py`](21-claude-agents-sdk-bedrock/claude-agent-agents-as-tools.py) | [`strands-agent-agents-as-tools.py`](31-strands-agents-sdk-bedrock/strands-agent-agents-as-tools.py) |
| [Handoffs](#8-handoffs) | [`openai-agent-handoffs.py`](11-openai-agents-sdk-bedrock/openai-agent-handoffs.py) | [`claude-agent-handoffs.py`](21-claude-agents-sdk-bedrock/claude-agent-handoffs.py) | [`strands-agent-handoffs.py`](31-strands-agents-sdk-bedrock/strands-agent-handoffs.py) |
| [Structured output](#9-structured-output) | [`openai-agent-structured-output.py`](11-openai-agents-sdk-bedrock/openai-agent-structured-output.py) | [`claude-agent-structured-output.py`](21-claude-agents-sdk-bedrock/claude-agent-structured-output.py) | [`strands-agent-structured-output.py`](31-strands-agents-sdk-bedrock/strands-agent-structured-output.py) |
| Guardrails | [`openai-agent-guardrails.py`](11-openai-agents-sdk-bedrock/openai-agent-guardrails.py) | [`claude-agent-guardrails.py`](21-claude-agents-sdk-bedrock/claude-agent-guardrails.py) | not yet |
| Deep research | [`openai-agent-deep-research.py`](11-openai-agents-sdk-bedrock/openai-agent-deep-research.py) | [`claude-agent-deep-research.py`](21-claude-agents-sdk-bedrock/claude-agent-deep-research.py) | not yet |

Guardrails and deep research have no Strands version yet, so they don't have a section here yet.

## 1. Basic agent

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Connect to Bedrock.** OpenAI registers a Bedrock client as the default for every agent. Claude sets an environment variable for the CLI it starts. Strands gives the model a boto3 session.

</td></tr>
<tr><td>

```python
set_tracing_disabled(True)
set_default_openai_client(
    AsyncOpenAI(
        provider=bedrock(region=region)
    )
)
```

</td><td>

```python
env={
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "AWS_PROFILE": profile,
    "AWS_REGION": region,
}
```

</td><td>

```python
model = BedrockModel(
    model_id=model_id,
    boto_session=boto3.Session(
        profile_name=profile,
        region_name=region,
    ),
)
```

</td></tr>

<tr><td colspan="3">

**Define the agent.** OpenAI and Strands have an `Agent` class. Claude has no agent object: the agent is a set of options for one run, including the Bedrock `env` above and turning off Claude Code's built-in tools and local settings.

</td></tr>
<tr><td>

```python
agent = Agent(
    name="Agent Tutor",
    instructions="You explain ...",
    model=model_id,
)
```

</td><td>

```python
options = ClaudeAgentOptions(
    model=model_id,
    system_prompt="You explain ...",
    tools=[],
    max_turns=1,
    setting_sources=[],
    env=env,
)
```

</td><td>

```python
agent = Agent(
    model=model,
    system_prompt="You explain ...",
    callback_handler=None,
)
```

</td></tr>

<tr><td colspan="3">

**Run it.** OpenAI awaits a single result. Strands calls the agent like a function, synchronously. Claude's `query()` only streams back messages from a Claude Code CLI subprocess and has no call that collects them, so a list comprehension reads them all. The last one is always a `ResultMessage` that sums up the run.

</td></tr>
<tr><td>

```python
result = await Runner.run(agent, prompt)
```

</td><td>

```python
messages = [
    message async for message in query(
        prompt=prompt,
        options=options,
    )
]
result = messages[-1]  # ResultMessage
```

</td><td>

```python
result = agent(prompt)
```

</td></tr>

<tr><td colspan="3">

**Get the answer.** Each SDK puts the final text on its result object.

</td></tr>
<tr><td>

```python
result.final_output
```

</td><td>

```python
result.result
```

</td><td>

```python
str(result)
```

</td></tr>

<tr><td colspan="3">

**Run summary.** Only Claude reports a cost. The others report token usage.

</td></tr>
<tr><td>

```python
result.context_wrapper.usage
```

</td><td>

```python
result.num_turns
result.duration_ms
result.total_cost_usd
```

</td><td>

```python
result.metrics.cycle_count
result.metrics.accumulated_usage
```

</td></tr>
</table>

## 2. Streaming

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Turn streaming on.** OpenAI has a separate run method. Claude turns it on with an option, and `query()` then also yields stream events. Strands has `stream_async()`. Its model always streams from Bedrock (ConverseStream), and `agent(prompt)` just waits for the end.

</td></tr>
<tr><td>

```python
result = Runner.run_streamed(
    agent, input=prompt
)
```

</td><td>

```python
options = ClaudeAgentOptions(
    ...,
    include_partial_messages=True,
)
```

</td><td>

```python
events = agent.stream_async(prompt)
```

</td></tr>

<tr><td colspan="3">

**Print the text as it arrives.** OpenAI and Claude yield raw model stream events. In Claude, the complete `AssistantMessage` still follows the events, so it is skipped to avoid printing the answer twice. Strands yields plain dicts: a text chunk has a `"data"` key, and the last event has a `"result"` key with the same `AgentResult` that `agent(prompt)` returns.

</td></tr>
<tr><td>

```python
events = result.stream_events()
async for event in events:
    data = event.data
    if (
        event.type == "raw_response_event"
        and isinstance(
            data, ResponseTextDeltaEvent
        )
    ):
        print(data.delta, end="")
```

</td><td>

```python
async for message in query(
    prompt=prompt,
    options=options,
):
    if isinstance(message, StreamEvent):
        e = message.event
        d = e.get("delta", {})
        if d.get("type") == "text_delta":
            print(d["text"], end="")
```

</td><td>

```python
async for event in events:
    if "data" in event:
        print(event["data"], end="")
    elif "result" in event:
        result = event["result"]
```

</td></tr>
</table>

## 3. Tools

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Declare a tool.** OpenAI and Strands read the name and parameters from the function signature and docstring. Claude takes them explicitly, with `Annotated` descriptions, and passes all arguments in one dict.

</td></tr>
<tr><td>

```python
@function_tool
def calculate_loan_payment(
    principal: float,
    annual_rate_percent: float,
    years: int,
) -> str:
    """Calculate the monthly payment
    and total interest for a loan.

    Args:
        principal: Amount borrowed.
        ...
    """
```

</td><td>

```python
@tool(
    "calculate_loan_payment",
    "Calculate the monthly payment "
    "and total interest for a loan.",
    {
        "principal": Annotated[
            float, "Amount borrowed."
        ],
        ...
    },
)
async def calculate_loan_payment(args):
```

</td><td>

```python
@tool
def calculate_loan_payment(
    principal: float,
    annual_rate_percent: float,
    years: int,
) -> str:
    """Calculate the monthly payment
    and total interest for a loan.

    Args:
        principal: Amount borrowed.
        ...
    """
```

</td></tr>

<tr><td colspan="3">

**Return the result.** OpenAI and Strands tools return a plain value. Claude tools return MCP-style content blocks.

</td></tr>
<tr><td>

```python
return f"Monthly payment ${payment}"
```

</td><td>

```python
return {"content": [{
    "type": "text",
    "text": f"Monthly payment ${payment}",
}]}
```

</td><td>

```python
return f"Monthly payment ${payment}"
```

</td></tr>

<tr><td colspan="3">

**Give the tools to the agent.** OpenAI and Strands take the functions directly. Claude serves the tools from an in-process MCP server, and the agent sees each one as `mcp__<server>__<tool>`. It also needs permission to run them, which the others don't.

</td></tr>
<tr><td>

```python
agent = Agent(
    ...,
    tools=[
        calculate_loan_payment,
        months_to_savings_goal,
    ],
)
```

</td><td>

```python
options = ClaudeAgentOptions(
    ...,
    mcp_servers={
        "finance": create_sdk_mcp_server(
            name="finance", tools=tools
        )
    },
    allowed_tools=[
        f"mcp__finance__{t.name}"
        for t in tools
    ],
    max_turns=5,
)
```

</td><td>

```python
agent = Agent(
    ...,
    tools=[
        calculate_loan_payment,
        months_to_savings_goal,
    ],
)
```

</td></tr>

<tr><td colspan="3">

**See which tools were called.** OpenAI lists calls and outputs as run items. Claude sends calls in assistant messages and results in user messages. Strands keeps both in `agent.messages`, as `toolUse` and `toolResult` blocks. All three match each call to its result by ID.

</td></tr>
<tr><td>

```python
for item in result.new_items:
    raw = item.raw_item
    if item.type == "tool_call_item":
        raw.name, raw.arguments
    elif item.type == "tool_call_output_item":
        raw["call_id"], item.output
```

</td><td>

```python
for msg in messages:
    if isinstance(msg, AssistantMessage):
        for b in msg.content:
            if isinstance(b, ToolUseBlock):
                b.name, b.input
    elif isinstance(msg, UserMessage):
        for b in msg.content:
            if isinstance(b, ToolResultBlock):
                b.tool_use_id, b.content
```

</td><td>

```python
for msg in agent.messages:
    for b in msg["content"]:
        if "toolUse" in b:
            u = b["toolUse"]
            u["name"], u["input"]
        elif "toolResult" in b:
            r = b["toolResult"]
            r["toolUseId"], r["content"]
```

</td></tr>
</table>

## 4. Conversation history

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Where the history lives.** In OpenAI you hold it. Each run is stateless, and `to_input_list()` gives you the conversation to pass back. In Claude, a `ClaudeSDKClient` runs one CLI process, and that process remembers. In Strands, the `Agent` object keeps it in `agent.messages`.

</td></tr>
<tr><td>

```python
history = response.to_input_list()
```

</td><td>

```python
async with ClaudeSDKClient(
    options=options
) as client:
    ...
```

</td><td>

```python
agent.messages  # Converse format
```

</td></tr>

<tr><td colspan="3">

**Continue the conversation.** OpenAI appends the new message to the history and runs again. Claude and Strands just ask the same client or agent again.

</td></tr>
<tr><td>

```python
response = await Runner.run(
    agent,
    history + [{
        "role": "user",
        "content": follow_up,
    }],
)
```

</td><td>

```python
await client.query(follow_up)
async for message in (
    client.receive_response()
):
    ...
```

</td><td>

```python
result = agent(follow_up)
```

</td></tr>

<tr><td colspan="3">

**Start fresh.** In OpenAI, run without the history. In Claude, use another client. In Strands, use another agent. A Strands `BedrockModel` holds no history, so the two agents can share one.

</td></tr>
<tr><td>

```python
await Runner.run(agent, follow_up)
```

</td><td>

```python
async with ClaudeSDKClient(
    options=options
) as fresh_client:
    ...
```

</td><td>

```python
fresh_agent = Agent(model=model, ...)
fresh_agent(follow_up)
```

</td></tr>
</table>

## 5. Sessions

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Attach a session.** OpenAI passes a session to each run and stores it in SQLite. Claude names the session in the options, and the CLI saves it as a transcript under `~/.claude/projects/`. Strands gives the agent a session manager, which writes each message to a JSON file as it is added.

</td></tr>
<tr><td>

```python
session = SQLiteSession(
    SESSION_ID, DB_PATH
)
await Runner.run(
    agent, prompt, session=session
)
```

</td><td>

```python
options = ClaudeAgentOptions(
    ...,
    session_id=SESSION_ID,
)
```

</td><td>

```python
agent = Agent(
    ...,
    session_manager=FileSessionManager(
        session_id=SESSION_ID,
        storage_dir=str(SESSIONS_DIR),
    ),
    agent_id=AGENT_ID,
)
```

</td></tr>

<tr><td colspan="3">

**Pick it up after a restart.** OpenAI reopens the session by ID and file. Claude starts a new client with `resume`. Strands builds a new agent with the same IDs, which loads the saved messages before its first call. None of them saves the system prompt; it comes from the code each time.

</td></tr>
<tr><td>

```python
session = SQLiteSession(
    SESSION_ID, DB_PATH
)
```

</td><td>

```python
options = ClaudeAgentOptions(
    ...,
    resume=SESSION_ID,
)
```

</td><td>

```python
agent = make_agent(model)
agent.messages  # already reloaded
```

</td></tr>

<tr><td colspan="3">

**Clean up.**

</td></tr>
<tr><td>

```python
await session.clear_session()
```

</td><td>

```python
delete_session(SESSION_ID)
```

</td><td>

```python
FileSessionManager(
    session_id=SESSION_ID,
    storage_dir=str(SESSIONS_DIR),
).delete_session(SESSION_ID)
```

</td></tr>
</table>

## 6. Workflow orchestrated by code

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Run the planners in parallel.** Each SDK has an awaitable run, so `asyncio.gather` runs them at once. In Claude, each run is its own `query()`, so its own CLI process. The Claude and Strands scripts wrap the call in a small `run_agent` helper.

</td></tr>
<tr><td>

```python
results = await asyncio.gather(*(
    Runner.run(planner, request)
    for planner in planners
))
```

</td><td>

```python
drafts = await asyncio.gather(*(
    run_agent(
        p, request, options[p.name]
    )
    for p in planners
))
```

</td><td>

```python
# run_agent awaits
# agent.invoke_async(prompt)
drafts = await asyncio.gather(*(
    run_agent(planner, request)
    for planner in planners
))
```

</td></tr>

<tr><td colspan="3">

**Pass results along.** The agents never see each other's conversations. The code passes each result on as the next agent's prompt.

</td></tr>
<tr><td>

```python
best = await Runner.run(
    editor, combined
)
await Runner.run(
    publisher, best.final_output
)
```

</td><td>

```python
best = await run_agent(
    editor, combined, options[...]
)
await run_agent(
    publisher, best.result.result, ...
)
```

</td><td>

```python
best = await run_agent(
    editor, combined
)
await run_agent(publisher, best)
```

</td></tr>
</table>

## 7. Agents as tools

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Make an agent callable.** OpenAI and Strands wrap an agent as a tool with `as_tool()`. Claude defines subagents, which the director reaches through Claude Code's built-in `Agent` tool.

</td></tr>
<tr><td>

```python
planner.as_tool(
    tool_name="nature_guide",
    tool_description=description,
)
```

</td><td>

```python
AgentDefinition(
    description=description,
    prompt=prompt,
    tools=[],
    model="inherit",
)
```

</td><td>

```python
planner.as_tool(
    name="nature_guide",
    description=description,
)
```

</td></tr>

<tr><td colspan="3">

**Give them to the director.** In Claude, subagents run in the background unless background tasks are turned off, and the director may move on without their results. Usage also differs: Claude's `total_cost_usd` includes the subagents, while Strands counts each planner's tokens on the planner.

</td></tr>
<tr><td>

```python
director = Agent(
    ...,
    tools=[
        *planner_tools, save_itinerary
    ],
)
```

</td><td>

```python
options = ClaudeAgentOptions(
    ...,
    agents=subagents,
    tools=["Agent"],
    allowed_tools=["Agent", save_tool],
    env={
        ...,
        "CLAUDE_CODE_DISABLE_"
        "BACKGROUND_TASKS": "1",
    },
)
```

</td><td>

```python
director = Agent(
    ...,
    tools=[
        *planner_tools, save_itinerary
    ],
)
```

</td></tr>
</table>

## 8. Handoffs

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Declare the handoff.** OpenAI has `handoffs=`. Claude has no handoff, so the example adds a tool that does no work: calling it is the signal. Strands puts the agents in a `Swarm`, which gives each one a `handoff_to_agent` tool.

</td></tr>
<tr><td>

```python
director = Agent(
    ...,
    tools=planner_tools,
    handoffs=[publisher],
)
```

</td><td>

```python
@tool(
    "transfer_to_publisher", ...,
)
async def transfer_to_publisher(args):
    return {"content": [...]}
```

</td><td>

```python
swarm = Swarm(
    [director, publisher],
    entry_point=director,
    max_handoffs=4,
)
```

</td></tr>

<tr><td colspan="3">

**Hand over.** OpenAI switches agents inside one run. In Claude, the code sees the tool call and runs the publisher with `resume`, on the director's session. Strands' swarm starts the next agent itself.

</td></tr>
<tr><td>

```python
result = await Runner.run(
    director, task
)
```

</td><td>

```python
director_run = await run_agent(
    task, director_options
)
await run_agent(
    "You are now Publisher ...",
    make_options(
        publisher, ...,
        resume=session_id,
    ),
)
```

</td><td>

```python
result = swarm(task)
```

</td></tr>

<tr><td colspan="3">

**What the next agent sees.** In OpenAI and Claude, the publisher gets the whole conversation, drafts included. In Strands, it gets only a prompt the swarm builds from the handoff message and the task, so the director must put the chosen plan in the message.

</td></tr>
<tr><td>

```python
result.last_agent
```

</td><td>

```python
publisher_run.result.result
```

</td><td>

```python
result.node_history
result.results[name].result
```

</td></tr>
</table>

## 9. Structured output

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Ask for a typed answer.** OpenAI and Strands take a Pydantic class. Claude has no `output_type`, so its example makes the schema a tool's input and reads the object off the call. It also shows the simpler way: asking for JSON in the system prompt.

</td></tr>
<tr><td>

```python
reviewer = Agent(
    ...,
    output_type=ItineraryReview,
)
```

</td><td>

```python
@tool(
    "submit_review", ...,
    ItineraryReview.model_json_schema(),
)
async def submit_review(args):
    ...
```

</td><td>

```python
reviewer = Agent(
    ...,
    structured_output_model=(
        ItineraryReview
    ),
)
```

</td></tr>

<tr><td colspan="3">

**Get the object.** Strands does Claude's tool approach for you: it adds a tool named after the class. In both, a call that doesn't match the schema goes back to the agent to fix, checked with jsonschema in Claude and Pydantic in Strands.

</td></tr>
<tr><td>

```python
review = result.final_output
```

</td><td>

```python
review = ItineraryReview.model_validate(
    submissions[-1].input
)
```

</td><td>

```python
review = result.structured_output
```

</td></tr>
</table>
