import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated
from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolResultBlock, ToolUseBlock, UserMessage,
    create_sdk_mcp_server, delete_session, query, tool,
)

OUTPUT_DIR = Path(__file__).with_name("itineraries") / "claude"
SERVER_NAME = "travel"


# --- Tool: used by the publisher in the last step ---

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


# --- Agents: the Claude Agent SDK has no Agent class, so an agent here is a name, a system prompt
#     and its tools, turned into ClaudeAgentOptions for each run ---

@dataclass
class Agent:
    name: str
    system_prompt: str
    tools: list = field(default_factory=list)  # @tool functions, served by an in-process MCP server


@dataclass
class Run:
    result: ResultMessage
    tool_calls: list[tuple[ToolUseBlock, str]]  # each call with the text it returned


def make_options(agent: Agent, model_id: str, profile: str, region: str) -> ClaudeAgentOptions:
    """Factory: the ClaudeAgentOptions for one agent, with its tools (if any) pre-approved."""
    tool_options = {}
    if agent.tools:
        tool_options = {
            "mcp_servers": {SERVER_NAME: create_sdk_mcp_server(name=SERVER_NAME, tools=agent.tools)},
            "allowed_tools": [f"mcp__{SERVER_NAME}__{t.name}" for t in agent.tools],
        }
    return ClaudeAgentOptions(
        model=model_id,
        system_prompt=agent.system_prompt,
        tools=[],  # no built-in tools (file access, shell, ...)
        max_turns=3 if agent.tools else 1,  # a tool call and its reply take extra turns
        setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
        env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region},
        **tool_options,
    )


def result_text(block: ToolResultBlock) -> str:
    """A tool result's content is either a string or a list of content blocks."""
    if isinstance(block.content, list):
        return " ".join(part.get("text", "") for part in block.content)
    return block.content or "(no result)"


async def run_agent(agent: Agent, prompt: str, options: ClaudeAgentOptions) -> Run:
    """Run one agent on one prompt (its own query(), so its own CLI process) and collect its tool calls."""
    calls: list[ToolUseBlock] = []
    outputs: dict[str, str] = {}
    final = None
    # Read to the end rather than returning at the ResultMessage, so query() can shut its CLI process down cleanly
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            calls += [block for block in message.content if isinstance(block, ToolUseBlock)]
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            outputs |= {b.tool_use_id: result_text(b) for b in message.content if isinstance(b, ToolResultBlock)}
        elif isinstance(message, ResultMessage):
            final = message
    return Run(final, [(call, outputs.get(call.id, "(no result)")) for call in calls])


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

    # 2. Three planners with the same job but different styles, so their drafts differ
    format_rule = "Give a morning, afternoon and evening, one line each, naming real places."
    planners = [
        Agent("Nature Guide", f"You plan Tokyo days around gardens, parks and easy hikes. {format_rule}"),
        Agent("Culture Guide", f"You plan Tokyo days around quiet shrines, temples and old neighborhoods. {format_rule}"),
        Agent("Slow Travel Guide", f"You plan unhurried Tokyo days with long walks, cafés and hot baths. {format_rule}"),
    ]
    editor = Agent(
        "Travel Editor",
        "You compare day plans for a traveler and pick the single best one for their request. "
        "Reply with the chosen plan exactly as written, followed by one line starting 'Why:'.",
    )
    publisher = Agent(
        "Publisher",
        "You turn a chosen day plan into a clean Markdown itinerary with a short title "
        "and save it with the save_itinerary tool. Reply with where it was saved.",
        tools=[save_itinerary],
    )
    options = {agent.name: make_options(agent, model_id, profile, region) for agent in [*planners, editor, publisher]}

    print("--- Agents ---")
    for agent in [*planners, editor, publisher]:
        tools = ", ".join(f"mcp__{SERVER_NAME}__{t.name}" for t in agent.tools) or "none"
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
    drafts = await asyncio.gather(*(run_agent(p, request, options[p.name]) for p in planners))
    print(f"({len(drafts)} drafts in {time.perf_counter() - start:.1f}s, run in parallel)\n")
    for planner, draft in zip(planners, drafts):
        print(f"[{planner.name}]\n{draft.result.result}\n")

    # 4. Step 2: combine the drafts into one prompt and let the editor pick
    print("=== Step 2: picking the best draft ===")
    combined = f"Traveler's request: {request}\n\n" + "\n\n".join(
        f"Plan {i} ({planner.name}):\n{draft.result.result}"
        for i, (planner, draft) in enumerate(zip(planners, drafts), 1)
    )
    best = await run_agent(editor, combined, options[editor.name])
    print(f"{best.result.result}\n")

    # 5. Step 3: hand the winner to the publisher, which calls the tool
    print("=== Step 3: publishing ===")
    published = await run_agent(publisher, best.result.result, options[publisher.name])
    for call, output in published.tool_calls:
        print(f"Called {call.name}(title={call.input.get('title')!r}, "
              f"markdown=<{len(call.input.get('markdown', ''))} chars>)")
        print(f"Result: {output}")
    print(f"\n{published.result.result}")

    # 6. Each agent ran as its own query(), so add up the runs for the whole workflow
    runs = [*drafts, best, published]
    print("\n--- Run Summary ---")
    print(f"Agent runs: {len(runs)}")
    print(f"Duration:   {time.perf_counter() - start:.1f}s")
    print(f"Cost:       ${sum(run.result.total_cost_usd or 0 for run in runs):.4f}")

    # 7. Clean up: each query() saved its session as a transcript under ~/.claude/projects/.
    #    The workflow never resumes them, so delete_session() removes them.
    for run in runs:
        delete_session(run.result.session_id)
    print(f"\n--- {len(runs)} agent sessions deleted (delete_session) ---")

if __name__ == "__main__":
    # Model output contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
