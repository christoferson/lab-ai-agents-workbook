import asyncio
import math
import os
import sys
from typing import Annotated, get_args
from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolResultBlock, ToolUseBlock, UserMessage,
    create_sdk_mcp_server, query, tool,
)


# --- Tools: plain Python functions the agent can decide to call ---
# LLMs are unreliable at compound-interest math, so the agent hands the numbers to these tools.
# @tool takes the name, the description and the input schema; Annotated adds a description to each parameter.
# A tool returns MCP-style content: a list of blocks, here a single text block.

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
    principal, years = args["principal"], args["years"]
    months = years * 12
    monthly_rate = args["annual_rate_percent"] / 100 / 12
    if monthly_rate == 0:
        payment = principal / months
    else:
        payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** -months)
    total_interest = payment * months - principal
    text = f"Monthly payment ${payment:,.2f} over {months} months; total interest ${total_interest:,.2f}"
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "months_to_savings_goal",
    "Calculate how many months of fixed deposits it takes to reach a savings goal, with monthly compounding.",
    {
        "goal": Annotated[float, "Target amount to save, e.g. 30000."],
        "monthly_deposit": Annotated[float, "Amount deposited at the end of each month."],
        "annual_rate_percent": Annotated[float, "Annual interest rate in percent earned on the savings, e.g. 4."],
    },
)
async def months_to_savings_goal(args):
    goal, monthly_deposit = args["goal"], args["monthly_deposit"]
    monthly_rate = args["annual_rate_percent"] / 100 / 12
    if monthly_rate == 0:
        months = math.ceil(goal / monthly_deposit)
        balance = monthly_deposit * months
    else:
        months = math.ceil(math.log(goal * monthly_rate / monthly_deposit + 1) / math.log(1 + monthly_rate))
        balance = monthly_deposit * ((1 + monthly_rate) ** months - 1) / monthly_rate
    deposited = monthly_deposit * months
    text = (f"{months} months ({months // 12} years {months % 12} months) to reach ${balance:,.2f}; "
            f"you deposit ${deposited:,.2f} and earn ${balance - deposited:,.2f} in interest")
    return {"content": [{"type": "text", "text": text}]}


def result_text(block: ToolResultBlock) -> str:
    """A tool result's content is either a string or a list of content blocks."""
    if isinstance(block.content, list):
        return " ".join(part.get("text", "") for part in block.content)
    return block.content or "(no result)"


async def main():

    # 1. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    model_id = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", "global.anthropic.claude-sonnet-5")

    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print(f"Claude Model ID:    {model_id}")
    print("--------------------------\n")

    # 2. Custom tools reach the agent through an MCP server. create_sdk_mcp_server runs it inside this
    #    Python process, and the agent sees each tool as mcp__<server name>__<tool name>.
    tools = [calculate_loan_payment, months_to_savings_goal]
    server_name = "finance"
    finance_server = create_sdk_mcp_server(name=server_name, tools=tools)
    tool_names = [f"mcp__{server_name}__{t.name}" for t in tools]

    system_prompt = (
        "You help people make purchase decisions. Always use your tools for calculations; "
        "never do the math yourself. Finish with a short recommendation."
    )
    options = ClaudeAgentOptions(
        model=model_id,
        system_prompt=system_prompt,
        tools=[],  # no built-in tools (file access, shell, ...); the MCP tools below are separate
        mcp_servers={server_name: finance_server},
        # Tools need permission to run. There is no one to ask in a script, so pre-approve our own tools.
        allowed_tools=tool_names,
        max_turns=5,  # each round of tool calls takes a turn, so allow a few
        setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
        env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region},
    )

    print("--- Agent ---")
    print(f"System prompt: {system_prompt}\n")

    print(f"--- Tools available to the agent (MCP server {server_name!r}) ---")
    for t, full_name in zip(tools, tool_names):
        print(f"* {full_name}: {t.description}")
        for param, annotated in t.input_schema.items():
            param_type, description = get_args(annotated)
            print(f"    - {param} ({param_type.__name__}): {description}")
    print()

    # 3. Run the agentic loop; the agent decides which tools to call and with what arguments
    prompt = (
        "I want to buy a $30,000 car. Option A: a 5-year loan at 6.5%. "
        "Option B: save $500 a month in an account paying 4% and buy it in cash. "
        "Compare the two options."
    )
    print("--- Prompt ---")
    print(f"{prompt}\n")

    print(f"Running agent on {model_id} via Amazon Bedrock...\n")
    # Tool calls arrive as ToolUseBlocks in AssistantMessages, and their results come back as
    # ToolResultBlocks in UserMessages, so collect both and match them by tool_use_id
    calls: list[ToolUseBlock] = []
    outputs: dict[str, str] = {}
    final = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            calls += [block for block in message.content if isinstance(block, ToolUseBlock)]
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            outputs |= {b.tool_use_id: result_text(b) for b in message.content if isinstance(b, ToolResultBlock)}
        elif isinstance(message, ResultMessage):
            final = message

    # Walk through what happened during the run: each tool call and what it returned
    print("--- Agent steps ---")
    for step, call in enumerate(calls, 1):
        args = ", ".join(f"{k}={v}" for k, v in call.input.items())
        print(f"{step}. Called {call.name}({args})")
        print(f"   Result: {outputs.get(call.id, '(no result)')}")
    if not calls:
        print("(no tools called)")

    print("\n--- Agent Response ---")
    print(final.result)

    print("\n--- Run Summary ---")
    print(f"Status:   {'error' if final.is_error else 'success'}")
    print(f"Turns:    {final.num_turns}")
    print(f"Duration: {final.duration_ms / 1000:.1f}s")
    if final.total_cost_usd is not None:
        print(f"Cost:     ${final.total_cost_usd:.4f}")

if __name__ == "__main__":
    # Model output contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
