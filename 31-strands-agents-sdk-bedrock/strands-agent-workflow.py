import asyncio
import os
import re
import sys
import time
from pathlib import Path
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

OUTPUT_DIR = Path(__file__).with_name("itineraries") / "strands"


# --- Tool: used by the publisher in the last step ---

@tool
def save_itinerary(title: str, markdown: str) -> str:
    """Save a finished itinerary as a Markdown file and return the file path.

    Args:
        title: Short itinerary title, e.g. 'A Quiet Spring Day in Tokyo'.
        markdown: The full itinerary in Markdown.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "itinerary"
    path = OUTPUT_DIR / f"{slug}.md"
    # Plans on the same topic often get the same title, so number new files instead of overwriting old ones
    n = 2
    while path.exists():
        path = OUTPUT_DIR / f"{slug}-{n}.md"
        n += 1
    # The agent often starts the plan with its own H1, so only add one when it's missing
    body = markdown.strip() if markdown.lstrip().startswith("# ") else f"# {title}\n\n{markdown.strip()}"
    path.write_text(body + "\n", encoding="utf-8")
    return f"Saved to {path.relative_to(Path(__file__).parent.parent).as_posix()}"


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


async def run_agent(agent: Agent, prompt: str) -> str:
    """Run one agent on one prompt and return its reply. invoke_async() is the awaitable form of
    agent(prompt), so several agents can run at once with asyncio.gather."""
    return str(await agent.invoke_async(prompt)).strip()


async def main():

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

    # 2. Five agents on one shared model. Each Agent keeps its own conversation, so they never see
    #    each other's messages; the code passes results along as prompts.
    model = BedrockModel(
        model_id=model_id,
        boto_session=boto3.Session(profile_name=profile, region_name=region),
    )

    def make_agent(name: str, system_prompt: str, tools: list | None = None) -> Agent:
        return Agent(name=name, model=model, system_prompt=system_prompt, tools=tools, callback_handler=None)

    # Three planners with the same job but different styles, so their drafts differ
    format_rule = "Give a morning, afternoon and evening, one line each, naming real places."
    planners = [
        make_agent("Nature Guide", f"You plan Tokyo days around gardens, parks and easy hikes. {format_rule}"),
        make_agent("Culture Guide", f"You plan Tokyo days around quiet shrines, temples and old neighborhoods. {format_rule}"),
        make_agent("Slow Travel Guide", f"You plan unhurried Tokyo days with long walks, cafés and hot baths. {format_rule}"),
    ]
    editor = make_agent(
        "Travel Editor",
        "You compare day plans for a traveler and pick the single best one for their request. "
        "Reply with the chosen plan exactly as written, followed by one line starting 'Why:'.",
    )
    publisher = make_agent(
        "Publisher",
        "You turn a chosen day plan into a clean Markdown itinerary with a short title "
        "and save it with the save_itinerary tool. Reply with where it was saved.",
        tools=[save_itinerary],
    )
    agents = [*planners, editor, publisher]

    print("--- Agents ---")
    for agent in agents:
        tools = ", ".join(agent.tool_names) or "none"
        print(f"* {agent.name} (tools: {tools})\n    {agent.system_prompt}")
    print()

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

    # 3. Step 1: run the planners concurrently; the code, not an agent, decides the flow
    print(f"=== Step 1: drafting on {model_id} via Amazon Bedrock ===")
    start = time.perf_counter()
    drafts = await asyncio.gather(*(run_agent(planner, request) for planner in planners))
    print(f"({len(drafts)} drafts in {time.perf_counter() - start:.1f}s, run in parallel)\n")
    for planner, draft in zip(planners, drafts):
        print(f"[{planner.name}]\n{draft}\n")

    # 4. Step 2: combine the drafts into one prompt and let the editor pick
    print("=== Step 2: picking the best draft ===")
    combined = f"Traveler's request: {request}\n\n" + "\n\n".join(
        f"Plan {i} ({planner.name}):\n{draft}" for i, (planner, draft) in enumerate(zip(planners, drafts), 1)
    )
    best = await run_agent(editor, combined)
    print(f"{best}\n")

    # 5. Step 3: hand the winner to the publisher, which calls the tool
    print("=== Step 3: publishing ===")
    published = await run_agent(publisher, best)
    for call, output in tool_steps(publisher.messages):
        print(f"Called {call['name']}(title={call['input'].get('title')!r}, "
              f"markdown=<{len(call['input'].get('markdown', ''))} chars>)")
        print(f"Result: {output}")
    print(f"\n{published}")

    # 6. Each agent tracks its own usage in event_loop_metrics, so add them up for the whole workflow
    usages = [agent.event_loop_metrics.accumulated_usage for agent in agents]
    print("\n--- Run Summary ---")
    print(f"Agent runs: {len(agents)}")
    print(f"Duration:   {time.perf_counter() - start:.1f}s")
    print(f"Tokens:     {sum(u['inputTokens'] for u in usages)} in, {sum(u['outputTokens'] for u in usages)} out")

if __name__ == "__main__":
    # Model output contains characters like en dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
