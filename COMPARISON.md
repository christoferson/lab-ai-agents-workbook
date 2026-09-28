# SDK comparison

This page compares the same examples in the OpenAI Agents SDK, the Claude Agent SDK and the Strands Agents SDK, all running on Amazon Bedrock. Each section shows only the code that differs. Environment printing, prompts and output formatting are left out, so follow the links for the full scripts.

| Example | OpenAI Agents SDK | Claude Agent SDK | Strands Agents SDK |
| --- | --- | --- | --- |
| [Basic agent](#1-basic-agent) | [`openai-agent.py`](11-openai-agents-sdk-bedrock/openai-agent.py) | [`claude-agent.py`](21-claude-agents-sdk-bedrock/claude-agent.py) | [`strands-agent.py`](31-strands-agents-sdk-bedrock/strands-agent.py) |
| [Streaming](#2-streaming) | [`openai-agent-streaming.py`](11-openai-agents-sdk-bedrock/openai-agent-streaming.py) | [`claude-agent-streaming.py`](21-claude-agents-sdk-bedrock/claude-agent-streaming.py) | not yet |
| [Tools](#3-tools) | [`openai-agent-tools.py`](11-openai-agents-sdk-bedrock/openai-agent-tools.py) | [`claude-agent-tools.py`](21-claude-agents-sdk-bedrock/claude-agent-tools.py) | not yet |

## 1. Basic agent

One question, one answer.

| | OpenAI Agents SDK | Claude Agent SDK | Strands Agents SDK |
| --- | --- | --- | --- |
| Connects to Bedrock by | Registering an `AsyncOpenAI` client with a `bedrock` provider as the default | Setting `CLAUDE_CODE_USE_BEDROCK=1` in the environment of the CLI it starts | A `BedrockModel` with a boto3 session |
| An agent is | `Agent(name, instructions, model)` | `ClaudeAgentOptions(model, system_prompt, ...)` passed to `query()` | `Agent(model, system_prompt)` |
| Runs as | `await Runner.run(agent, prompt)` | `async for message in query(...)`, which runs the bundled Claude Code CLI as a subprocess | `agent(prompt)` (synchronous) |
| Final answer | `result.final_output` | `ResultMessage.result` | `str(result)` |
| Run summary | Token usage in `result.raw_responses` | Turns, duration and cost (USD) on `ResultMessage` | Cycles and token usage on `result.metrics` |
| Things to turn off | Tracing: `set_tracing_disabled(True)` | Built-in tools (`tools=[]`) and local settings (`setting_sources=[]`) | Printing while the response streams: `callback_handler=None` |

**OpenAI Agents SDK**

```python
from openai import AsyncOpenAI
from openai.providers import bedrock
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled

set_tracing_disabled(True)  # no OpenAI tracing endpoint to export to
set_default_openai_client(AsyncOpenAI(provider=bedrock(region=region)))

agent = Agent(
    name="Agent Tutor",
    instructions="You explain AI agent concepts clearly and concisely to developers.",
    model=model_id,  # openai.gpt-5.6-luna
)
result = await Runner.run(agent, prompt)
print(result.final_output)
```

**Claude Agent SDK**

```python
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

options = ClaudeAgentOptions(
    model=model_id,  # global.anthropic.claude-sonnet-5
    system_prompt="You explain AI agent concepts clearly and concisely to developers.",
    tools=[],            # no built-in tools (file access, shell, ...)
    max_turns=1,
    setting_sources=[],  # don't load ~/.claude settings or CLAUDE.md files
    env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region},
)
async for message in query(prompt=prompt, options=options):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)
    elif isinstance(message, ResultMessage):
        print(message.num_turns, message.duration_ms, message.total_cost_usd)
```

**Strands Agents SDK**

```python
import boto3
from strands import Agent
from strands.models.bedrock import BedrockModel

model = BedrockModel(
    model_id=model_id,  # Claude, gpt-oss or Nova, chosen by STRANDS_MODEL_PROVIDER
    boto_session=boto3.Session(profile_name=profile, region_name=region),
)
agent = Agent(
    model=model,
    system_prompt="You explain AI agent concepts clearly and concisely to developers.",
    callback_handler=None,  # don't print the response while it streams
)
result = agent(prompt)  # synchronous
print(str(result))
print(result.metrics.cycle_count, result.metrics.accumulated_usage)
```

## 2. Streaming

Print the answer as it is generated. The setup is the same as in the basic agent; only the run changes.

| | OpenAI Agents SDK | Claude Agent SDK | Strands Agents SDK |
| --- | --- | --- | --- |
| Turn streaming on | Call `Runner.run_streamed(...)` instead of `Runner.run(...)` | `include_partial_messages=True` | not yet |
| Text arrives as | `raw_response_event` whose data is a `ResponseTextDeltaEvent` | `StreamEvent` whose event is a `content_block_delta` with a `text_delta` | not yet |
| Full message too? | No, only events | Yes, the complete `AssistantMessage` still follows, so skip it to avoid printing twice | not yet |

**OpenAI Agents SDK**

```python
from openai.types.responses import ResponseTextDeltaEvent

result = Runner.run_streamed(agent, input=prompt)
async for event in result.stream_events():
    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        print(event.data.delta, end="", flush=True)
```

**Claude Agent SDK**

```python
from claude_agent_sdk import StreamEvent

options = ClaudeAgentOptions(..., include_partial_messages=True)

async for message in query(prompt=prompt, options=options):
    if isinstance(message, StreamEvent):
        event = message.event
        if event.get("type") == "content_block_delta" and event["delta"].get("type") == "text_delta":
            print(event["delta"]["text"], end="", flush=True)
```

**Strands Agents SDK**

No example yet.

## 3. Tools

The agent calls Python functions for the loan and savings math. The math is the same in every version; only how the tool is declared, registered and reported differs.

| | OpenAI Agents SDK | Claude Agent SDK | Strands Agents SDK |
| --- | --- | --- | --- |
| Declare a tool | `@function_tool` on a normal function | `@tool(name, description, schema)` on an `async def f(args)` | not yet |
| Describe the parameters | Docstring `Args:` section | `Annotated[type, "description"]` in the schema | not yet |
| Tool returns | A plain value (here a string) | MCP content: `{"content": [{"type": "text", "text": ...}]}` | not yet |
| Give it to the agent | `Agent(tools=[...])` | An in-process MCP server: `create_sdk_mcp_server(...)`, passed as `mcp_servers={...}` | not yet |
| Name the agent sees | `calculate_loan_payment` | `mcp__finance__calculate_loan_payment` | not yet |
| Permission to run | Not needed | Required: `allowed_tools=[...]` | not yet |
| See the calls | `result.new_items`: `tool_call_item` and `tool_call_output_item`, matched by `call_id` | `ToolUseBlock` in `AssistantMessage` and `ToolResultBlock` in `UserMessage`, matched by `tool_use_id` | not yet |

**OpenAI Agents SDK**

```python
from agents import Agent, Runner, function_tool

@function_tool
def calculate_loan_payment(principal: float, annual_rate_percent: float, years: int) -> str:
    """Calculate the monthly payment and total interest for a fixed-rate loan.

    Args:
        principal: Amount borrowed, e.g. 30000.
        annual_rate_percent: Annual interest rate in percent, e.g. 6.5.
        years: Loan term in years.
    """
    ...
    return f"Monthly payment ${payment:,.2f} over {months} months; ..."

agent = Agent(name="Finance Assistant", instructions=..., model=model_id,
              tools=[calculate_loan_payment, months_to_savings_goal])
result = await Runner.run(agent, prompt)

outputs = {item.raw_item["call_id"]: item.output
           for item in result.new_items if item.type == "tool_call_output_item"}
for item in result.new_items:
    if item.type == "tool_call_item":
        print(item.raw_item.name, item.raw_item.arguments, outputs[item.raw_item.call_id])
```

**Claude Agent SDK**

```python
from typing import Annotated
from claude_agent_sdk import ToolResultBlock, ToolUseBlock, UserMessage, create_sdk_mcp_server, tool

@tool(
    "calculate_loan_payment",
    "Calculate the monthly payment and total interest for a fixed-rate loan.",
    {
        "principal": Annotated[float, "Amount borrowed, e.g. 30000."],
        "annual_rate_percent": Annotated[float, "Annual interest rate in percent, e.g. 6.5."],
        "years": Annotated[int, "Loan term in years."],
    },
)
async def calculate_loan_payment(args):
    ...
    return {"content": [{"type": "text", "text": f"Monthly payment ${payment:,.2f} ..."}]}

tools = [calculate_loan_payment, months_to_savings_goal]
options = ClaudeAgentOptions(
    ...,
    tools=[],  # built-in tools stay off; MCP tools are separate
    mcp_servers={"finance": create_sdk_mcp_server(name="finance", tools=tools)},
    allowed_tools=[f"mcp__finance__{t.name}" for t in tools],  # pre-approve them
    max_turns=5,  # each round of tool calls takes a turn
)

async for message in query(prompt=prompt, options=options):
    if isinstance(message, AssistantMessage):
        calls += [b for b in message.content if isinstance(b, ToolUseBlock)]
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        outputs |= {b.tool_use_id: b.content for b in message.content if isinstance(b, ToolResultBlock)}
```

**Strands Agents SDK**

No example yet.
