import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Annotated
from claude_agent_sdk import (
    AgentDefinition, AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolResultBlock, ToolUseBlock,
    UserMessage, create_sdk_mcp_server, delete_session, query, tool,
)

OUTPUT_DIR = Path(__file__).with_name("itineraries") / "claude"
SERVER_NAME = "travel"
AGENT_TOOL = "Agent"  # Claude Code's built-in tool for calling a subagent


# --- Tool: the director calls this once it has picked a plan ---

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


def result_text(block: ToolResultBlock) -> str:
    """A tool result's text. The CLI wraps a subagent's report in a notice and a footer (its agent ID and
    usage) meant for the model, so keep only the report itself."""
    content = block.content
    text = " ".join(part.get("text", "") for part in content) if isinstance(content, list) else content or ""
    if "The report follows:\n" in text:
        report = text.split("The report follows:\n", 1)[1].split("\nagentId:", 1)[0]
        text = "\n".join(line.removeprefix("  ") for line in report.splitlines())  # the CLI indents each line
    return text.strip() or "(no result)"


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

    # 2. The same three planners as the code-orchestrated workflow, defined as subagents. The description
    #    tells the director when to use one; the prompt is the subagent's own system prompt.
    format_rule = "Give a morning, afternoon and evening, one line each, naming real places."
    # The director writes each subagent's prompt itself, so ask it to pass just the request and not
    # add format instructions of its own, which would override the planner's one-line format.
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

    # 3. The director: the LLM, not your code, decides which subagents and tools to call and in what order.
    #    It reaches the subagents through the built-in Agent tool, and save_itinerary through an MCP server.
    save_tool = f"mcp__{SERVER_NAME}__{save_itinerary.name}"
    system_prompt = (
        "You are a trip director. Your goal is to deliver the single best day plan "
        "for the traveler using your planner subagents."
    )
    options = ClaudeAgentOptions(
        model=model_id,
        system_prompt=system_prompt,
        agents=subagents,
        tools=[AGENT_TOOL],  # of the built-in tools, only the one that calls subagents
        mcp_servers={SERVER_NAME: create_sdk_mcp_server(name=SERVER_NAME, tools=[save_itinerary])},
        allowed_tools=[AGENT_TOOL, save_tool],  # pre-approve both, as there is no one to ask
        max_turns=8,
        setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
        env={
            "CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region,
            # Subagents run in the background by default, and their drafts would arrive later as
            # notifications. Turn that off so each Agent call waits and returns the draft as its result.
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        },
    )

    print("--- Agent graph ---")
    print(f"Trip Director: {system_prompt}")
    for name, prompt in planners.items():
        print(f"  |- {name} (subagent, via the {AGENT_TOOL} tool): {prompt}")
    print(f"  |- {save_tool} (MCP tool): {save_itinerary.description}\n")

    task = """
Traveler's request: one relaxed day in Tokyo in April for someone who loves nature and wants to avoid crowds.

Follow these steps:
1. Generate drafts: call each of the three planner subagents once with the traveler's request.
   Do not continue until you have all three drafts.
2. Evaluate and select: pick the single best plan for this traveler.
3. Save only the best plan with save_itinerary as clean Markdown with a short title.
Finally, reply with which planner won, why in one sentence, and where the plan was saved.
""".strip()
    print("--- Task ---")
    print(f"{task}\n")

    print(f"Running Trip Director on {model_id} via Amazon Bedrock...\n")
    # The director's tool calls arrive as ToolUseBlocks and their results as ToolResultBlocks, matched by
    # tool_use_id. Messages from inside a subagent carry its parent_tool_use_id, so skip those.
    calls: list[ToolUseBlock] = []
    outputs: dict[str, str] = {}
    final = None
    async for message in query(prompt=task, options=options):
        if isinstance(message, AssistantMessage) and message.parent_tool_use_id is None:
            calls += [block for block in message.content if isinstance(block, ToolUseBlock)]
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            outputs |= {b.tool_use_id: result_text(b) for b in message.content if isinstance(b, ToolResultBlock)}
        elif isinstance(message, ResultMessage):
            final = message

    # Walk through the decisions the director made: each call and what came back
    print("--- Director's steps ---")
    for step, call in enumerate(calls, 1):
        if call.name == AGENT_TOOL:
            shown = f"subagent_type={call.input.get('subagent_type')!r}, prompt={call.input.get('prompt')!r}"
        else:
            shown = f"title={call.input.get('title')!r}, markdown=<{len(call.input.get('markdown', ''))} chars>"
        print(f"{step}. Called {call.name}({shown})")
        print("   Result: " + outputs.get(call.id, "(no result)").replace("\n", "\n           ") + "\n")

    print("--- Director's Response ---")
    print(final.result)

    print("\n--- Run Summary ---")
    print(f"Status:   {'error' if final.is_error else 'success'}")
    print(f"Turns:    {final.num_turns}")  # the director's turns; each subagent runs its own
    print(f"Duration: {final.duration_ms / 1000:.1f}s")
    if final.total_cost_usd is not None:
        print(f"Cost:     ${final.total_cost_usd:.4f}")  # includes the subagents

    # 4. Clean up: delete_session() removes the saved transcript, including the subagents' transcripts
    delete_session(final.session_id)
    print(f"\n--- Session deleted (delete_session): {final.session_id} ---")

if __name__ == "__main__":
    # Model output contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
