import asyncio
import json
import math
import os
from openai import AsyncOpenAI
from openai.providers import bedrock
# Import the explicit tracing control from the agents SDK
from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_disabled


# --- Tools: plain Python functions the agent can decide to call ---
# LLMs are unreliable at compound-interest math, so the agent hands the numbers to these tools.
# The docstring summary becomes the tool description and the Args section documents each parameter.

@function_tool
def calculate_loan_payment(principal: float, annual_rate_percent: float, years: int) -> str:
    """Calculate the monthly payment and total interest for a fixed-rate loan.

    Args:
        principal: Amount borrowed, e.g. 30000.
        annual_rate_percent: Annual interest rate in percent, e.g. 6.5.
        years: Loan term in years.
    """
    months = years * 12
    monthly_rate = annual_rate_percent / 100 / 12
    if monthly_rate == 0:
        payment = principal / months
    else:
        payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** -months)
    total_interest = payment * months - principal
    return f"Monthly payment ${payment:,.2f} over {months} months; total interest ${total_interest:,.2f}"


@function_tool
def months_to_savings_goal(goal: float, monthly_deposit: float, annual_rate_percent: float) -> str:
    """Calculate how many months of fixed deposits it takes to reach a savings goal, with monthly compounding.

    Args:
        goal: Target amount to save, e.g. 30000.
        monthly_deposit: Amount deposited at the end of each month.
        annual_rate_percent: Annual interest rate in percent earned on the savings, e.g. 4.
    """
    monthly_rate = annual_rate_percent / 100 / 12
    if monthly_rate == 0:
        months = math.ceil(goal / monthly_deposit)
        balance = monthly_deposit * months
    else:
        months = math.ceil(math.log(goal * monthly_rate / monthly_deposit + 1) / math.log(1 + monthly_rate))
        balance = monthly_deposit * ((1 + monthly_rate) ** months - 1) / monthly_rate
    deposited = monthly_deposit * months
    return (f"{months} months ({months // 12} years {months % 12} months) to reach ${balance:,.2f}; "
            f"you deposit ${deposited:,.2f} and earn ${balance - deposited:,.2f} in interest")


async def main():

    # 1. Disable the telemetry trace exporter directly via the SDK function
    set_tracing_disabled(True)

    # 2. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    model_id = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-5.6-luna")

    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print(f"Bedrock Model ID:   {model_id}")
    print("--------------------------\n")

    # 3. Instantiate Bedrock client using the official provider framework.
    bedrock_client = AsyncOpenAI(
        provider=bedrock(
            region=region
        )
    )

    # 4. Register your Bedrock client as default for the OpenAI Agents SDK
    set_default_openai_client(bedrock_client)

    # 5. Initialize your Agent with the tools it may call
    tools = [calculate_loan_payment, months_to_savings_goal]
    agent = Agent(
        name="Finance Assistant",
        instructions=(
            "You help people make purchase decisions. Always use your tools for calculations; "
            "never do the math yourself. Finish with a short recommendation."
        ),
        model=model_id,
        tools=tools,
    )

    print("--- Agent ---")
    print(f"Name:         {agent.name}")
    print(f"Instructions: {agent.instructions}\n")

    print("--- Tools available to the agent ---")
    for tool in tools:
        print(f"* {tool.name}: {tool.description}")
        for param, schema in tool.params_json_schema["properties"].items():
            print(f"    - {param} ({schema['type']}): {schema.get('description', '')}")
    print()

    # 6. Run the agentic loop; the agent decides which tools to call and with what arguments
    prompt = (
        "I want to buy a $30,000 car. Option A: a 5-year loan at 6.5%. "
        "Option B: save $500 a month in an account paying 4% and buy it in cash. "
        "Compare the two options."
    )
    print("--- Prompt ---")
    print(f"{prompt}\n")

    print(f"Running agent on {model_id} via Amazon Bedrock...\n")
    result = await Runner.run(agent, prompt)

    # Walk through what happened during the run: each tool call and what it returned
    print("--- Agent steps ---")
    step = 0
    for item in result.new_items:
        if item.type == "tool_call_item":
            step += 1
            args = ", ".join(f"{k}={v}" for k, v in json.loads(item.raw_item.arguments).items())
            print(f"{step}. Called {item.raw_item.name}({args})")
        elif item.type == "tool_call_output_item":
            print(f"   Result: {item.output}")
    if step == 0:
        print("(no tools called)")

    print("\n--- Agent Response ---")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
