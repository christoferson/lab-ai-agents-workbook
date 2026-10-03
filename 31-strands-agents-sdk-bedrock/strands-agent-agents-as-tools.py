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


# --- Tool: the director calls this once it has picked a plan ---

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

    model = BedrockModel(
        model_id=model_id,
        boto_session=boto3.Session(profile_name=profile, region_name=region),
    )

    def make_agent(name: str, system_prompt: str, tools: list | None = None) -> Agent:
        return Agent(name=name, model=model, system_prompt=system_prompt, tools=tools, callback_handler=None)

    # 2. The same three planners as the code-orchestrated workflow
    format_rule = "Give a morning, afternoon and evening, one line each, naming real places."
    planners = [
        make_agent("Nature Guide", f"You plan Tokyo days around gardens, parks and easy hikes. {format_rule}"),
        make_agent("Culture Guide", f"You plan Tokyo days around quiet shrines, temples and old neighborhoods. {format_rule}"),
        make_agent("Slow Travel Guide", f"You plan unhurried Tokyo days with long walks, cafés and hot baths. {format_rule}"),
    ]

    # 3. as_tool() wraps each agent as a tool with one string parameter, "input": calling it runs that
    #    agent on the input and returns its reply. Control comes back to the caller afterwards
    #    (Director -> Planner -> Director). Each call starts the planner from a fresh conversation.
    description = "Drafts a one-day Tokyo plan. In the input, describe the traveler's request."
    planner_tools = [
        planner.as_tool(name=planner.name.lower().replace(" ", "_"), description=description)  # names can't have spaces
        for planner in planners
    ]

    # 4. The director: the LLM, not your code, decides which tools to call and in what order
    director = make_agent(
        "Trip Director",
        "You are a trip director. Your goal is to deliver the single best day plan "
        "for the traveler using your planner tools.",
        tools=[*planner_tools, save_itinerary],
    )

    print("--- Agent graph ---")
    print(f"{director.name}: {director.system_prompt}")
    for t in director.tool_registry.registry.values():
        print(f"  |- {t.tool_name} ({t.tool_type} tool): {t.tool_spec['description']}")
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
    start = time.perf_counter()
    result = director(task)
    duration = time.perf_counter() - start

    # 5. Walk through the decisions the director made: each tool call and what came back.
    #    Only the director's calls are in director.messages; each planner's run stays in its own agent.
    print("--- Director's steps ---")
    for step, (call, output) in enumerate(tool_steps(director.messages), 1):
        args = call["input"]
        if call["name"] == save_itinerary.tool_name:
            shown = f"title={args.get('title')!r}, markdown=<{len(args.get('markdown', ''))} chars>"
        else:
            shown = f"input={args.get('input')!r}"
        print(f"{step}. Called {call['name']}({shown})")
        print("   Result: " + output.strip().replace("\n", "\n           ") + "\n")

    print("--- Director's Response ---")
    print(str(result).strip())

    # 6. Each agent counts only its own model calls, so the planners' tokens are reported separately
    def tokens(agents: list[Agent]) -> str:
        usages = [agent.event_loop_metrics.accumulated_usage for agent in agents]
        return f"{sum(u['inputTokens'] for u in usages)} in, {sum(u['outputTokens'] for u in usages)} out"

    print("\n--- Run Summary ---")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Cycles:      {result.metrics.cycle_count}")  # the director's model calls; each planner runs its own
    print(f"Duration:    {duration:.1f}s")
    print(f"Tokens:      director {tokens([director])}; planners {tokens(planners)}")

if __name__ == "__main__":
    # Model output contains characters like en dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    main()
