import asyncio
import json
import os
import re
import time
from pathlib import Path
from openai import AsyncOpenAI
from openai.providers import bedrock
# Import the explicit tracing control from the agents SDK
from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_disabled

OUTPUT_DIR = Path(__file__).with_name("itineraries")


# --- Tool: used by the publisher in the last step ---

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

    # 5. Three planners with the same job but different styles, so their drafts differ
    format_rule = "Give a morning, afternoon and evening, one line each, naming real places."
    planners = [
        Agent(name="Nature Guide", model=model_id,
              instructions=f"You plan Tokyo days around gardens, parks and easy hikes. {format_rule}"),
        Agent(name="Culture Guide", model=model_id,
              instructions=f"You plan Tokyo days around quiet shrines, temples and old neighborhoods. {format_rule}"),
        Agent(name="Slow Travel Guide", model=model_id,
              instructions=f"You plan unhurried Tokyo days with long walks, cafés and hot baths. {format_rule}"),
    ]

    editor = Agent(
        name="Travel Editor", model=model_id,
        instructions=(
            "You compare day plans for a traveler and pick the single best one for their request. "
            "Reply with the chosen plan exactly as written, followed by one line starting 'Why:'."
        ),
    )

    publisher = Agent(
        name="Publisher", model=model_id,
        instructions=(
            "You turn a chosen day plan into a clean Markdown itinerary with a short title "
            "and save it with the save_itinerary tool. Reply with where it was saved."
        ),
        tools=[save_itinerary],
    )

    print("--- Workflow ---")
    print(f"Step 1: {', '.join(p.name for p in planners)} draft plans in parallel")
    print(f"Step 2: {editor.name} picks the best draft")
    print(f"Step 3: {publisher.name} formats it and saves it with the save_itinerary tool\n")

    request = (
        "Plan one relaxed day in Tokyo in April for a traveler who loves nature "
        "and wants to avoid crowds."
    )
    print("--- Request ---")
    print(f"{request}\n")

    # 6. Step 1: run the planners concurrently; the code, not an agent, decides the flow
    print(f"=== Step 1: drafting on {model_id} via Amazon Bedrock ===")
    start = time.perf_counter()
    results = await asyncio.gather(*(Runner.run(planner, request) for planner in planners))
    print(f"(3 drafts in {time.perf_counter() - start:.1f}s, run in parallel)\n")

    drafts = [result.final_output for result in results]
    for planner, draft in zip(planners, drafts):
        print(f"[{planner.name}]\n{draft}\n")

    # 7. Step 2: combine the drafts into one input and let the editor pick
    print("=== Step 2: picking the best draft ===")
    combined = f"Traveler's request: {request}\n\n" + "\n\n".join(
        f"Plan {i} ({planner.name}):\n{draft}" for i, (planner, draft) in enumerate(zip(planners, drafts), 1)
    )
    best = await Runner.run(editor, combined)
    print(f"{best.final_output}\n")

    # 8. Step 3: hand the winner to the publisher, which calls the tool
    print("=== Step 3: publishing ===")
    published = await Runner.run(publisher, best.final_output)
    for item in published.new_items:
        if item.type == "tool_call_item":
            args = json.loads(item.raw_item.arguments)
            print(f"Called {item.raw_item.name}(title={args.get('title')!r}, markdown=<{len(args.get('markdown', ''))} chars>)")
        elif item.type == "tool_call_output_item":
            print(f"Result: {item.output}")
    print(f"\n{published.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
