import asyncio
import json
import os
import re
from pathlib import Path
from openai import AsyncOpenAI
from openai.providers import bedrock
# Import the explicit tracing control from the agents SDK
from agents import Agent, Runner, function_tool, handoff, set_default_openai_client, set_tracing_disabled

OUTPUT_DIR = Path(__file__).with_name("itineraries")


# --- Function tool: only the Publisher has this ---

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

    # 5. The same three planners, used as tools (Director -> Planner -> Director)
    format_rule = "Give a morning, afternoon and evening, one line each, naming real places."
    planners = [
        Agent(name="Nature Guide", model=model_id,
              instructions=f"You plan Tokyo days around gardens, parks and easy hikes. {format_rule}"),
        Agent(name="Culture Guide", model=model_id,
              instructions=f"You plan Tokyo days around quiet shrines, temples and old neighborhoods. {format_rule}"),
        Agent(name="Slow Travel Guide", model=model_id,
              instructions=f"You plan unhurried Tokyo days with long walks, cafés and hot baths. {format_rule}"),
    ]
    description = "Drafts a one-day Tokyo plan. In the input, describe the traveler's request."
    tools = [
        planner.as_tool(tool_name=planner.name.lower().replace(" ", "_"), tool_description=description)
        for planner in planners
    ]

    # 6. The Publisher is a handoff target, not a tool. Once the Director hands off,
    #    the Publisher owns the conversation and its reply is the final output.
    publisher = Agent(
        name="Publisher",
        model=model_id,
        handoff_description="Formats a chosen day plan as Markdown, saves it and replies to the traveler.",
        instructions=(
            "You receive a conversation in which a day plan was chosen. Turn the chosen plan into clean "
            "Markdown with a short title, save it with save_itinerary, then reply to the traveler with "
            "a warm two-sentence summary and where the plan was saved."
        ),
        tools=[save_itinerary],
    )
    handoffs = [publisher]

    director = Agent(
        name="Trip Director",
        model=model_id,
        instructions=(
            "You are a trip director. Your goal is to find the single best day plan for the traveler "
            "using your planner tools, then hand off to the Publisher to save and deliver it."
        ),
        tools=tools,
        handoffs=handoffs,
    )

    print("--- Agent graph ---")
    print(director.name)
    for tool in director.tools:
        print(f"  |- {tool.name} (agent tool: returns to {director.name})")
    print(f"  `- {handoff(publisher).tool_name} (handoff: {publisher.name} takes over)")
    for tool in publisher.tools:
        print(f"       `- {tool.name} (function tool)")
    print()

    task = """
Traveler's request: one relaxed day in Tokyo in April for someone who loves nature and wants to avoid crowds.

Follow these steps:
1. Generate drafts: call each of the three planner tools once with the traveler's request.
   Do not continue until you have all three drafts.
2. Evaluate and select: pick the single best plan and state which planner wrote it and why, in one sentence.
3. Hand off to the Publisher with only the best plan.
""".strip()
    print("--- Task ---")
    print(f"{task}\n")

    print(f"Running {director.name} on {model_id} via Amazon Bedrock...\n")
    result = await Runner.run(director, task)

    # Tool calls can run in parallel, so match each result to its call by call_id
    outputs = {
        item.raw_item["call_id"]: item.output
        for item in result.new_items if item.type == "tool_call_output_item"
    }

    # Walk through the run, showing which agent was in control at each step
    print("--- Steps ---")
    step = 0
    for item in result.new_items:
        if item.type == "tool_call_item":
            step += 1
            args = json.loads(item.raw_item.arguments)
            if item.raw_item.name == save_itinerary.name:
                shown = f"title={args.get('title')!r}, markdown=<{len(args.get('markdown', ''))} chars>"
            else:
                shown = f"input={args.get('input')!r}"
            print(f"{step}. [{item.agent.name}] called {item.raw_item.name}({shown})")
            result_text = str(outputs.get(item.raw_item.call_id, "(no result)"))
            print("   Result: " + result_text.replace("\n", "\n           ") + "\n")
        elif item.type == "handoff_output_item":
            step += 1
            print(f"{step}. [{item.source_agent.name}] handed off to {item.target_agent.name}\n")
        elif item.type == "message_output_item" and item.agent is director:
            # The Director's reasoning before handing off: which plan it picked and why
            print(f"   [{director.name}] said: " + "".join(
                part.text for part in item.raw_item.content if hasattr(part, "text")
            ) + "\n")

    print(f"--- Final Response (from {result.last_agent.name}) ---")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
