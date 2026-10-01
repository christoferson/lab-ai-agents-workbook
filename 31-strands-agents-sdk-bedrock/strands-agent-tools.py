import math
import os
import sys
import time
import boto3
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

# Which model family to use, and the env var holding its Bedrock model ID (plus its default).
# All are called through Bedrock's Converse API, so the same BedrockModel class serves each one.
PROVIDERS = {
    "anthropic": ("STRANDS_MODEL_ID_ANTHROPIC", "global.anthropic.claude-sonnet-5"),
    "openai": ("STRANDS_MODEL_ID_OPENAI", "openai.gpt-oss-120b-1:0"),  # OpenAI's open-weight gpt-oss
    "amazon": ("STRANDS_MODEL_ID_AMAZON", "global.amazon.nova-2-lite-v1:0"),  # Amazon's own Nova
}


# --- Tools: plain Python functions the agent can decide to call ---
# LLMs are unreliable at compound-interest math, so the agent hands the numbers to these tools.
# @tool reads everything from the function: the name, the type hints, the docstring's first line
# (the description) and its Args section (one description per parameter). A tool returns plain text.

@tool
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


@tool
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


def print_tool(t) -> None:
    """Print the spec Strands built from the function: what the model sees."""
    spec = t.tool_spec
    print(f"* {spec['name']}: {spec['description']}")
    for param, schema in spec["inputSchema"]["json"]["properties"].items():
        print(f"    - {param} ({schema['type']}): {schema['description']}")


def tool_steps(messages: list[dict]) -> list[tuple[dict, str]]:
    """Pair each toolUse block (in an assistant message) with its toolResult block (in the next
    user message), matched by toolUseId."""
    blocks = [block for message in messages for block in message["content"]]
    results = {
        block["toolResult"]["toolUseId"]: " ".join(part.get("text", "") for part in block["toolResult"]["content"])
        for block in blocks if "toolResult" in block
    }
    return [(block["toolUse"], results.get(block["toolUse"]["toolUseId"], "(no result)"))
            for block in blocks if "toolUse" in block]


def main():

    # 1. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    provider = os.environ.get("STRANDS_MODEL_PROVIDER", "anthropic")
    if provider not in PROVIDERS:
        sys.exit(f"STRANDS_MODEL_PROVIDER must be one of {', '.join(PROVIDERS)}, not {provider!r}")
    model_env, default_model_id = PROVIDERS[provider]
    model_id = os.environ.get(model_env, default_model_id)

    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print(f"Model Provider:     {provider} (STRANDS_MODEL_PROVIDER)")
    print(f"Model ID:           {model_id} ({model_env})")
    print("--------------------------\n")

    # 2. The tools go straight to the agent: no server and no permission list.
    #    Strands runs them in this process whenever the model asks for one.
    model = BedrockModel(
        model_id=model_id,
        boto_session=boto3.Session(profile_name=profile, region_name=region),
    )
    system_prompt = (
        "You help people make purchase decisions. Always use your tools for calculations; "
        "never do the math yourself. Finish with a short recommendation."
    )
    tools = [calculate_loan_payment, months_to_savings_goal]
    agent = Agent(model=model, system_prompt=system_prompt, tools=tools, callback_handler=None)

    print("--- Agent ---")
    print(f"System prompt: {system_prompt}\n")

    print("--- Tools available to the agent ---")
    for t in tools:
        print_tool(t)
    print()

    # 3. Run the agent loop; the model decides which tools to call and with what arguments
    prompt = (
        "I want to buy a $30,000 car. Option A: a 5-year loan at 6.5%. "
        "Option B: save $500 a month in an account paying 4% and buy it in cash. "
        "Compare the two options."
    )
    print("--- Prompt ---")
    print(f"{prompt}\n")

    print(f"Running agent on {model_id} via Amazon Bedrock...\n")
    start = time.perf_counter()
    result = agent(prompt)
    duration = time.perf_counter() - start

    # 4. agent.messages holds the whole conversation in Converse format. Tool calls are toolUse
    #    blocks in assistant messages, and their results are toolResult blocks in user messages.
    print("--- Agent steps ---")
    steps = tool_steps(agent.messages)
    for step, (call, output) in enumerate(steps, 1):
        args = ", ".join(f"{k}={v}" for k, v in call["input"].items())
        print(f"{step}. Called {call['name']}({args})")
        print(f"   Result: {output}")
    if not steps:
        print("(no tools called)")

    print("\n--- Agent Response ---")
    print(str(result).strip())

    # 5. Each cycle is one model call; a round of tool calls adds a cycle
    usage = result.metrics.accumulated_usage
    print("\n--- Run Summary ---")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Cycles:      {result.metrics.cycle_count}")
    print(f"Duration:    {duration:.1f}s")
    print(f"Tokens:      {usage['inputTokens']} in, {usage['outputTokens']} out")

if __name__ == "__main__":
    # Model output contains characters like en dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    main()
