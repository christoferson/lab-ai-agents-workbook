import asyncio
import json
import os
import re
from pathlib import Path
from openai import AsyncOpenAI
from openai.providers import bedrock
# Import the explicit tracing control from the agents SDK
from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_disabled

OUTPUT_DIR = Path(__file__).with_name("itineraries")


# --- Function tool: the director calls this once it has picked a plan ---

@function_tool
def save_itinerary(title: str, markdown: str) -> str:
    """Save a finished itinerary as a Markdown file and return the file path.

    Args:
        title: Short itinerary title, e.g. 'A Quiet Spring Day in Tokyo'.
        markdown: The full itinerary in Markdown.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "itinerary"
    path = OUTPUT_DIR / f"{slug}.md"
    path.write_text(f"# {title}\n\n{markdown}\n", encoding="utf-8")
    return f"Saved to {path.relative_to(Path(__file__).parent.parent)}"


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

    # 5. The same three planners as the code-orchestrated workflow
    format_rule = "Give a morning, afternoon and evening, one line each, naming real places."
    planners = [
        Agent(name="Nature Guide", model=model_id,
              instructions=f"You plan Tokyo days around gardens, parks and easy hikes. {format_rule}"),
        Agent(name="Culture Guide", model=model_id,
              instructions=f"You plan Tokyo days around quiet shrines, temples and old neighborhoods. {format_rule}"),
        Agent(name="Slow Travel Guide", model=model_id,
              instructions=f"You plan unhurried Tokyo days with long walks, cafés and hot baths. {format_rule}"),
    ]

    # 6. as_tool() wraps each agent as a tool: calling it runs that agent and returns its final output.
    #    Control comes back to the caller afterwards (Director -> Planner -> Director).
    description = "Drafts a one-day Tokyo plan. In the input, describe the traveler's request."
    tools = [
        planner.as_tool(tool_name=planner.name.lower().replace(" ", "_"), tool_description=description)
        for planner in planners
    ] + [save_itinerary]

    # 7. The planning agent: an LLM, not your code, decides which tools to call and in what order
    director = Agent(
        name="Trip Director",
        model=model_id,
        instructions=(
            "You are a trip director. Your goal is to deliver the single best day plan "
            "for the traveler using your planner tools."
        ),
        tools=tools,
    )

    print("--- Agent graph ---")
    print(f"{director.name}")
    for tool in director.tools:
        kind = "agent" if tool.name != save_itinerary.name else "function"
        print(f"  └─ {tool.name} ({kind} tool)")
    print()

    task = """
Traveler's request: one relaxed day in Tokyo in April for someone who loves nature and wants to avoid crowds.

Follow these steps:
1. Generate drafts: call each of the three planner tools once with the traveler's request.
   Do not continue until you have all three drafts.
2. Evaluate and select: pick the single best plan for this traveler.
3. Save only the best plan with save_itinerary as clean Markdown with a short title.
Finally, reply with which planner won, why in one sentence, and where the plan was saved.
""".strip()
    print("--- Task ---")
    print(f"{task}\n")

    print(f"Running {director.name} on {model_id} via Amazon Bedrock...\n")
    result = await Runner.run(director, task)

    # Walk through the decisions the director made: each tool call and what came back
    print("--- Director's steps ---")
    step = 0
    for item in result.new_items:
        if item.type == "tool_call_item":
            step += 1
            args = json.loads(item.raw_item.arguments)
            if item.raw_item.name == save_itinerary.name:
                shown = f"title={args.get('title')!r}, markdown=<{len(args.get('markdown', ''))} chars>"
            else:
                shown = f"input={args.get('input')!r}"
            print(f"{step}. Called {item.raw_item.name}({shown})")
        elif item.type == "tool_call_output_item":
            print("   Result: " + str(item.output).replace("\n", "\n           ") + "\n")

    print("--- Director's Response ---")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
