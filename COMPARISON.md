# SDK comparison

The same examples in the OpenAI Agents SDK, the Claude Agent SDK and the Strands Agents SDK, all running on Amazon Bedrock. Each aspect of an example has a divider row that explains how the SDKs differ, followed by a row with only the code for it in each SDK. Follow the links for the full scripts.

| Example | OpenAI Agents SDK | Claude Agent SDK | Strands Agents SDK |
| --- | --- | --- | --- |
| [Basic agent](#1-basic-agent) | [`openai-agent.py`](11-openai-agents-sdk-bedrock/openai-agent.py) | [`claude-agent.py`](21-claude-agents-sdk-bedrock/claude-agent.py) | [`strands-agent.py`](31-strands-agents-sdk-bedrock/strands-agent.py) |
| [Streaming](#2-streaming) | [`openai-agent-streaming.py`](11-openai-agents-sdk-bedrock/openai-agent-streaming.py) | [`claude-agent-streaming.py`](21-claude-agents-sdk-bedrock/claude-agent-streaming.py) | not yet |
| [Tools](#3-tools) | [`openai-agent-tools.py`](11-openai-agents-sdk-bedrock/openai-agent-tools.py) | [`claude-agent-tools.py`](21-claude-agents-sdk-bedrock/claude-agent-tools.py) | not yet |

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

**Turn streaming on.** OpenAI has a separate run method. Claude turns it on with an option, and `query()` then also yields stream events.

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

not yet

</td></tr>

<tr><td colspan="3">

**Print the text as it arrives.** Both yield raw model stream events. In Claude, the complete `AssistantMessage` still follows the events, so it is skipped to avoid printing the answer twice.

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

not yet

</td></tr>
</table>

## 3. Tools

<table>
<tr><th width="33%">OpenAI Agents SDK</th><th width="33%">Claude Agent SDK</th><th width="33%">Strands Agents SDK</th></tr>

<tr><td colspan="3">

**Declare a tool.** OpenAI reads the name and parameters from the function signature and docstring. Claude takes them explicitly, with `Annotated` descriptions, and passes all arguments in one dict.

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

not yet

</td></tr>

<tr><td colspan="3">

**Return the result.** Claude tools return MCP-style content blocks.

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

not yet

</td></tr>

<tr><td colspan="3">

**Give the tools to the agent.** Claude serves the tools from an in-process MCP server, and the agent sees each one as `mcp__<server>__<tool>`. It also needs permission to run them, which OpenAI doesn't.

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

not yet

</td></tr>

<tr><td colspan="3">

**See which tools were called.** OpenAI lists calls and outputs as run items. Claude sends calls in assistant messages and results in user messages. Both match each call to its result by ID.

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

not yet

</td></tr>
</table>
