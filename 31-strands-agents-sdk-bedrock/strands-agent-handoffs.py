import os
import re
import sys
import time
from pathlib import Path
import boto3
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.multiagent import Swarm

# Which model family to use, and the env var holding its Bedrock model ID (plus its default).
# All are called through Bedrock's Converse API, so the same BedrockModel class serves each one.
PROVIDERS = {
    "anthropic": ("STRANDS_MODEL_ID_ANTHROPIC", "global.anthropic.claude-sonnet-5"),
    "openai": ("STRANDS_MODEL_ID_OPENAI", "openai.gpt-oss-120b-1:0"),  # OpenAI's open-weight gpt-oss
    "amazon": ("STRANDS_MODEL_ID_AMAZON", "global.amazon.nova-2-lite-v1:0"),  # Amazon's own Nova
}

OUTPUT_DIR = Path(__file__).with_name("itineraries") / "strands"
HANDOFF_TOOL = "handoff_to_agent"  # the tool Swarm adds to every agent in it


# --- Tool: used by the publisher after it takes over ---

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


def indent(text: str) -> str:
    """Indent the lines after the first, so multi-line text lines up under its label."""
    return text.strip().replace("\n", "\n           ")


def print_step(step: int, agent_name: str, call: dict, output: str) -> None:
    """Print one tool call, labelled with the agent that made it."""
    args = call["input"]
    if call["name"] == HANDOFF_TOOL:
        shown = f"agent_name={args.get('agent_name')!r}, message=<{len(args.get('message', ''))} chars>, context={args.get('context')!r}"
    elif call["name"] == save_itinerary.tool_name:
        shown = f"title={args.get('title')!r}, markdown=<{len(args.get('markdown', ''))} chars>"
    else:
        shown = f"input={args.get('input')!r}"
    print(f"{step}. [{agent_name}] called {call['name']}({shown})")
    if call["name"] == HANDOFF_TOOL:
        # The result just echoes the message, which the next step shows inside the publisher's prompt
        output = output.splitlines()[0] + " ..."
    print(f"   Result: {indent(output)}\n")


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

    def make_agent(name: str, system_prompt: str, tools: list | None = None, description: str | None = None) -> Agent:
        return Agent(name=name, description=description, model=model, system_prompt=system_prompt,
                     tools=tools, callback_handler=None)

    # 2. The same three planners, used as tools (Director -> Planner -> Director). A tool is not a
    #    handoff: the director stays in control of the conversation.
    format_rule = "Give a morning, afternoon and evening, one line each, naming real places."
    planners = [
        make_agent("Nature Guide", f"You plan Tokyo days around gardens, parks and easy hikes. {format_rule}"),
        make_agent("Culture Guide", f"You plan Tokyo days around quiet shrines, temples and old neighborhoods. {format_rule}"),
        make_agent("Slow Travel Guide", f"You plan unhurried Tokyo days with long walks, cafés and hot baths. {format_rule}"),
    ]
    description = "Drafts a one-day Tokyo plan. In the input, describe the traveler's request."
    planner_tools = [
        planner.as_tool(name=planner.name.lower().replace(" ", "_"), description=description)
        for planner in planners
    ]

    # 3. The two agents that take turns owning the task. Their names are the swarm's node IDs, which
    #    is what the director passes to handoff_to_agent, so they are written like identifiers.
    publisher = make_agent(
        "publisher",
        "You have taken over a task in which a day plan was chosen. Turn that chosen plan into clean "
        "Markdown with a short title, save it with save_itinerary, then reply to the traveler with "
        "a warm two-sentence summary and where the plan was saved.",
        tools=[save_itinerary],
        description="Formats a chosen day plan as Markdown, saves it and replies to the traveler.",
    )
    # The steps live in the director's system prompt, not the task: Swarm repeats the task in every
    # agent's prompt, and steps written there made the publisher hand back to the director.
    director = make_agent(
        "trip_director",
        """You are a trip director. Find the single best day plan for the traveler, then hand it off.
1. Generate drafts: call each of the three planner tools once with the traveler's request.
   Do not continue until you have all three drafts.
2. Evaluate and select: pick the single best plan and state which planner wrote it and why, in one sentence.
3. Hand off to the publisher. The publisher sees only your handoff, not this conversation,
   so put the full text of the chosen plan in the handoff message.""",
        tools=planner_tools,
    )

    # 4. A Swarm is a group of agents that hand work to each other. It gives every agent a
    #    handoff_to_agent tool; calling it ends that agent's turn and starts the named one.
    swarm = Swarm([director, publisher], entry_point=director, max_handoffs=4)  # counts agent turns, not handoffs

    print("--- Agent graph (Swarm) ---")
    print(f"{director.name} (entry point)")
    for t in planner_tools:
        print(f"  |- {t.tool_name} (agent tool: returns to {director.name})")
    print(f"  `- {HANDOFF_TOOL}(agent_name={publisher.name!r}) (handoff: {publisher.name} takes over)")
    print(f"       `- {save_itinerary.tool_name} (function tool)")
    print()

    # 5. What the next agent receives. Swarm does not pass the conversation on: it builds the next
    #    agent's prompt from the handoff message, the original task and the handoff's context.
    print("--- How the handoff works ---")
    print(f"1. {director.name} calls {HANDOFF_TOOL}(agent_name, message, context), which ends its turn.")
    print(f"2. Swarm starts {publisher.name} with a new prompt: the message, the task and the context.")
    print(f"3. {publisher.name} never sees the drafts, so the message must carry the chosen plan.\n")

    task = "One relaxed day in Tokyo in April for someone who loves nature and wants to avoid crowds."
    print("--- Task ---")
    print(f"{task}\n")

    print(f"Running the swarm on {model_id} via Amazon Bedrock...\n")
    start = time.perf_counter()
    result = swarm(task)
    duration = time.perf_counter() - start

    # 6. node_history lists the agents in the order they had control. Swarm resets an agent's
    #    messages before each of its turns, so they hold only its last turn, starting with its prompt.
    print("--- Steps ---")
    step = 0
    history = [node.executor for node in result.node_history]
    for i, agent in enumerate(history):
        if agent in history[i + 1:]:
            step += 1
            print(f"{step}. [{agent.name}] had control (not shown: its messages now hold a later turn)\n")
            continue
        if agent is not director:
            step += 1
            prompt = "".join(block.get("text", "") for block in agent.messages[0]["content"])
            print(f"{step}. [{agent.name}] took over. The prompt Swarm built for it:")
            print("   | " + prompt.strip().replace("\n", "\n   | ") + "\n")
        for call, output in tool_steps(agent.messages):
            step += 1
            print_step(step, agent.name, call, output)

    last = result.node_history[-1].executor
    print(f"--- Final Response (from {last.name}) ---")
    print(str(result.results[last.name].result).strip())

    # 7. Each agent counts its own tokens across all its turns. The swarm's accumulated_usage adds up
    #    those running totals once per turn, so it overcounts an agent that gets control twice.
    def tokens(agents: list[Agent]) -> str:
        usages = [agent.event_loop_metrics.accumulated_usage for agent in agents]
        return f"{sum(u['inputTokens'] for u in usages)} in, {sum(u['outputTokens'] for u in usages)} out"

    print("\n--- Run Summary ---")
    print(f"Status:       {result.status.value}")
    print(f"Handoff path: {' -> '.join(node.node_id for node in result.node_history)}")
    print(f"Duration:     {duration:.1f}s")
    print(f"Tokens:       {director.name} {tokens([director])}; {publisher.name} {tokens([publisher])}; "
          f"planners {tokens(planners)}")

if __name__ == "__main__":
    # Model output contains characters like en dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    main()
