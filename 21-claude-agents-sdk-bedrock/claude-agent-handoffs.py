import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated
from claude_agent_sdk import (
    AgentDefinition, AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolResultBlock, ToolUseBlock,
    UserMessage, create_sdk_mcp_server, delete_session, query, tool,
)

OUTPUT_DIR = Path(__file__).with_name("itineraries") / "claude"
SERVER_NAME = "travel"
AGENT_TOOL = "Agent"  # Claude Code's built-in tool for calling a subagent


# --- Tools: the director hands off with one, the publisher saves with the other ---

@tool(
    "transfer_to_publisher",
    "Hand the conversation over to the Publisher once you have chosen the best plan. "
    "The Publisher can read this whole conversation, so do not repeat the plan.",
    {"note": Annotated[str, "One sentence: which planner you picked and why."]},
)
async def transfer_to_publisher(args):
    # A handoff tool does no work: calling it is the signal. The code below sees the call and
    # gives the conversation to the publisher, so this only tells the director it worked.
    return {"content": [{"type": "text", "text": f"Handing over to the Publisher. Your note: {args['note']}"}]}


@tool(
    "save_itinerary",
    "Save a finished itinerary as a Markdown file and return the file path.",
    {
        "title": Annotated[str, "Short itinerary title, e.g. 'A Quiet Spring Day in Tokyo'."],
        "markdown": Annotated[str, "The full itinerary in Markdown."],
    },
)
async def save_itinerary(args):
    title, markdown = args["title"], args["markdown"]
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
    text = f"Saved to {path.relative_to(Path(__file__).parent.parent).as_posix()}"
    return {"content": [{"type": "text", "text": text}]}


# --- Agents: as in the workflow example, an agent here is a name, a system prompt and what it can
#     call, turned into ClaudeAgentOptions for each run ---

@dataclass
class Agent:
    name: str
    system_prompt: str
    tools: list = field(default_factory=list)  # @tool functions, served by an in-process MCP server
    subagents: dict = field(default_factory=dict)  # name -> AgentDefinition, called via the Agent tool
    max_turns: int = 1


@dataclass
class Run:
    result: ResultMessage
    steps: list[tuple[ToolUseBlock, str]]  # each of this agent's own tool calls with the text it returned


def tool_name(sdk_tool) -> str:
    """The name the agent sees for an MCP tool: mcp__<server>__<tool>."""
    return f"mcp__{SERVER_NAME}__{sdk_tool.name}"


def make_options(agent: Agent, model_id: str, profile: str, region: str, resume: str | None = None) -> ClaudeAgentOptions:
    """Factory: the ClaudeAgentOptions for one agent. With resume=<session id>, the agent continues
    that session instead of starting fresh, which is what turns a handoff into a handoff."""
    built_in_tools = [AGENT_TOOL] if agent.subagents else []
    return ClaudeAgentOptions(
        model=model_id,
        system_prompt=agent.system_prompt,
        agents=agent.subagents,
        tools=built_in_tools,  # of the built-in tools, only the one that calls subagents
        mcp_servers={SERVER_NAME: create_sdk_mcp_server(name=SERVER_NAME, tools=agent.tools)},
        allowed_tools=[tool_name(t) for t in agent.tools] + built_in_tools,  # pre-approve; no one to ask
        max_turns=agent.max_turns,
        resume=resume,
        setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
        env={
            "CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region,
            # Subagents run in the background by default, and their drafts would arrive later as
            # notifications. Turn that off so each Agent call waits and returns the draft as its result.
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        },
    )


def result_text(block: ToolResultBlock) -> str:
    """A tool result's text. The CLI wraps a subagent's report in a notice and a footer (its agent ID and
    usage) meant for the model, so keep only the report itself."""
    content = block.content
    text = " ".join(part.get("text", "") for part in content) if isinstance(content, list) else content or ""
    if "The report follows:\n" in text:
        report = text.split("The report follows:\n", 1)[1].split("\nagentId:", 1)[0]
        text = "\n".join(line.removeprefix("  ") for line in report.splitlines())  # the CLI indents each line
    return text.strip() or "(no result)"


async def run_agent(prompt: str, options: ClaudeAgentOptions) -> Run:
    """Run one agent and collect its own tool calls. Messages from inside a subagent carry its
    parent_tool_use_id, so skip those: they are the subagent's steps, not this agent's."""
    calls: list[ToolUseBlock] = []
    outputs: dict[str, str] = {}
    final = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
            calls += [block for block in message.content if isinstance(block, ToolUseBlock)]
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            outputs |= {b.tool_use_id: result_text(b) for b in message.content if isinstance(b, ToolResultBlock)}
        elif isinstance(message, ResultMessage):
            final = message
    return Run(final, [(call, outputs.get(call.id, "(no result)")) for call in calls])


def print_step(step: int, agent_name: str, call: ToolUseBlock, output: str) -> None:
    """One line per call, with the agent that made it, then its result indented under it."""
    if call.name == AGENT_TOOL:
        shown = f"subagent_type={call.input.get('subagent_type')!r}, prompt={call.input.get('prompt')!r}"
    elif call.name == tool_name(save_itinerary):
        shown = f"title={call.input.get('title')!r}, markdown=<{len(call.input.get('markdown', ''))} chars>"
    else:
        shown = ", ".join(f"{key}={value!r}" for key, value in call.input.items())
    print(f"{step}. [{agent_name}] called {call.name}({shown})")
    print("   Result: " + output.replace("\n", "\n           ") + "\n")


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

    # 2. The same three planners as the agents-as-tools example: subagents that report back to the
    #    director. A subagent is not a handoff; the director stays in control of the conversation.
    format_rule = "Give a morning, afternoon and evening, one line each, naming real places."
    description = (
        "Drafts a short one-day Tokyo plan in its own format. "
        "In the prompt, pass only the traveler's request, without format instructions."
    )
    planners = {
        "nature_guide": f"You plan Tokyo days around gardens, parks and easy hikes. {format_rule}",
        "culture_guide": f"You plan Tokyo days around quiet shrines, temples and old neighborhoods. {format_rule}",
        "slow_travel_guide": f"You plan unhurried Tokyo days with long walks, cafés and hot baths. {format_rule}",
    }
    subagents = {
        name: AgentDefinition(description=description, prompt=prompt, tools=[], model="inherit")  # no tools; same model
        for name, prompt in planners.items()
    }

    # 3. The two agents that take turns owning the conversation
    director = Agent(
        name="Trip Director",
        system_prompt=(
            "You are a trip director. Your goal is to find the single best day plan for the traveler "
            "using your planner subagents, then hand off to the Publisher to save and deliver it."
        ),
        tools=[transfer_to_publisher],
        subagents=subagents,
        max_turns=8,  # three subagent calls, the handoff, and its own replies
    )
    publisher = Agent(
        name="Publisher",
        system_prompt=(
            "You have taken over a conversation in which a day plan was chosen. Turn that chosen plan into "
            "clean Markdown with a short title, save it with save_itinerary, then reply to the traveler with "
            "a warm two-sentence summary and where the plan was saved."
        ),
        tools=[save_itinerary],
        max_turns=3,  # a tool call and its reply take extra turns
    )

    print("--- Agent graph ---")
    print(director.name)
    for name in planners:
        print(f"  |- {name} (subagent via the {AGENT_TOOL} tool: reports back to {director.name})")
    print(f"  `- {tool_name(transfer_to_publisher)} (handoff: {publisher.name} takes over)")
    for sdk_tool in publisher.tools:
        print(f"       `- {tool_name(sdk_tool)} (MCP tool)")
    print()

    # 4. What a handoff has to mean, and how it is built here. The SDK has no handoff of its own:
    #    a session is the conversation, so resuming the director's session hands it to the publisher.
    print("--- How the handoff works ---")
    print(f"1. {director.name} calls {transfer_to_publisher.name}, which ends its run.")
    print(f"2. The code runs {publisher.name} with resume=<the director's session id>.")
    print(f"3. {publisher.name} therefore sees the whole conversation, and its reply is the final answer.\n")

    task = """
Traveler's request: one relaxed day in Tokyo in April for someone who loves nature and wants to avoid crowds.

Follow these steps:
1. Generate drafts: call each of the three planner subagents once with the traveler's request.
   Do not continue until you have all three drafts.
2. Evaluate and select: pick the single best plan for this traveler.
3. Hand off with transfer_to_publisher, noting which planner you picked and why.
""".strip()
    print("--- Task ---")
    print(f"{task}\n")

    # 5. Step 1: the director drafts and chooses, and ends by asking for the handoff
    print(f"Running {director.name} on {model_id} via Amazon Bedrock...\n")
    start = time.perf_counter()
    director_run = await run_agent(task, make_options(director, model_id, profile, region))
    session_id = director_run.result.session_id

    # 6. Step 2: the handoff. Only do it if the director actually asked for one.
    handed_off = any(call.name == tool_name(transfer_to_publisher) for call, _ in director_run.steps)
    publisher_run = None
    if handed_off:
        # query() always needs a prompt, so the handoff itself is the prompt. The publisher's
        # instructions and the conversation it just inherited tell it what to do.
        publisher_run = await run_agent(
            f"You are now {publisher.name} and this conversation is yours. Follow your instructions.",
            make_options(publisher, model_id, profile, region, resume=session_id),
        )

    # Walk through the run, showing which agent was in control at each step
    print("--- Steps ---")
    step = 0
    for call, output in director_run.steps:
        step += 1
        print_step(step, director.name, call, output)
    if director_run.result.result.strip():
        print(f"   [{director.name}] said: {director_run.result.result.strip()}\n")
    if publisher_run:
        step += 1
        print(f"{step}. [{publisher.name}] took over the conversation (resume={session_id})\n")
        for call, output in publisher_run.steps:
            step += 1
            print_step(step, publisher.name, call, output)

    last = publisher_run or director_run
    last_agent = publisher.name if publisher_run else f"{director.name}, which never handed off"
    print(f"--- Final Response (from {last_agent}) ---")
    print(last.result.result)

    # 7. Both agents ran as their own query(), so add up the runs for the whole handoff
    runs = [run for run in (director_run, publisher_run) if run]
    print("\n--- Run Summary ---")
    print(f"Agent runs: {len(runs)}")
    print(f"Turns:      {', '.join(str(run.result.num_turns) for run in runs)}")
    print(f"Duration:   {time.perf_counter() - start:.1f}s")
    print(f"Cost:       ${sum(run.result.total_cost_usd or 0 for run in runs):.4f}")  # includes the subagents

    # 8. Clean up: both runs share one session, since the second one resumed the first
    sessions = {run.result.session_id for run in runs}
    for sid in sessions:
        delete_session(sid)
    print(f"\n--- {len(sessions)} session deleted (delete_session): {', '.join(sessions)} ---")

if __name__ == "__main__":
    # Model output contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
