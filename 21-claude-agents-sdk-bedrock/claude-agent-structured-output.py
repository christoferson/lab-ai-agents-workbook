import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Literal
from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolUseBlock, create_sdk_mcp_server, delete_session,
    query, tool,
)
from pydantic import BaseModel, Field, ValidationError

SERVER_NAME = "review"


# --- The shape we want back. The Claude Agent SDK has no output_type, so this one Pydantic model is
#     used two ways below: as a schema in the prompt, and as a tool's input schema. ---
# Field descriptions are sent to the model either way, so they double as instructions for each field.

class ItineraryReview(BaseModel):
    is_realistic: bool = Field(description="Whether the plan can be done at a relaxed pace in one day")
    crowd_risk: Literal["low", "medium", "high"] = Field(description="How crowded the chosen places and times are likely to be")
    nature_score: int = Field(description="How nature-focused the plan is, from 1 (not at all) to 10 (entirely)")
    issues: list[str] = Field(description="Specific problems with the plan, one short sentence each")
    suggested_fix: str = Field(description="One concrete change that would improve the plan the most")


REVIEWER_ROLE = "You review Tokyo day plans for travelers who want a relaxed, nature-focused day away from crowds."


# --- Way 2: the same model as a tool. The agent fills in the tool's arguments instead of writing
#     text, and the SDK validates them against the schema before the handler runs. ---

@tool(
    "submit_review",
    "Submit your review of the itinerary. Call this exactly once, and reply with nothing else.",
    ItineraryReview.model_json_schema(),  # @tool takes a full JSON Schema, which Pydantic generates
)
async def submit_review(args):
    # The arguments are the review. The script reads them off the tool call, so this only confirms it.
    # A call that doesn't match the schema never gets here: the SDK checks it with jsonschema first and
    # returns the error to the agent, which can then try again.
    return {"content": [{"type": "text", "text": "Review recorded."}]}


# A deliberately bad plan: rushed, crowded, and barely any nature
SAMPLE_ITINERARY = """
Saturday in Tokyo, relaxed nature day:
- 8:00 Shibuya Crossing and Hachiko for photos
- 9:30 Senso-ji and Nakamise Street
- 11:00 Day trip to Mount Fuji 5th Station
- 15:00 Tokyo Disneyland
- 20:00 Robot show in Shinjuku, then karaoke until late
""".strip()


PLAN_FORMAT = "Reply with a title line and 5 or 6 timed bullet points only."

# Planner agents that can write a fresh itinerary to review: name -> system prompt
PLANNERS = {
    "chaotic": (
        "You write Tokyo day plans that claim to be a relaxed nature day but are really overpacked, "
        f"visit famous crowded spots at peak times, and contain almost no nature. {PLAN_FORMAT}"
    ),
    "thoughtful": (
        "You write genuinely relaxed, nature-focused Tokyo day plans: gardens, parks and easy hikes, "
        "few stops close together, quiet times of day, and realistic travel time between them. "
        f"{PLAN_FORMAT}"
    ),
    "offbeat": (
        "You write relaxed, nature-focused Tokyo day plans that go off the beaten path: lesser-known gardens, "
        "neighborhood shrines, river walks and green spaces most tourists skip, avoiding famous landmarks. "
        f"Name real places and add a few words on why each is special. {PLAN_FORMAT}"
    ),
}


@dataclass
class Run:
    result: ResultMessage
    calls: list[ToolUseBlock]


def make_options(system_prompt: str, model_id: str, profile: str, region: str,
                 tools: list = (), max_turns: int = 1) -> ClaudeAgentOptions:
    """Factory: the ClaudeAgentOptions for one run, with its tools (if any) pre-approved."""
    tool_options = {}
    if tools:
        tool_options = {
            "mcp_servers": {SERVER_NAME: create_sdk_mcp_server(name=SERVER_NAME, tools=list(tools))},
            "allowed_tools": [f"mcp__{SERVER_NAME}__{t.name}" for t in tools],
        }
    return ClaudeAgentOptions(
        model=model_id,
        system_prompt=system_prompt,
        tools=[],  # no built-in tools (file access, shell, ...)
        max_turns=max_turns,
        setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
        env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region},
        **tool_options,
    )


async def run_agent(prompt: str, options: ClaudeAgentOptions) -> Run:
    """Run one agent, keeping its final message and any tool calls it made."""
    calls: list[ToolUseBlock] = []
    final = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            calls += [block for block in message.content if isinstance(block, ToolUseBlock)]
        elif isinstance(message, ResultMessage):
            final = message
    return Run(final, calls)


def parse_json_reply(text: str) -> ItineraryReview:
    """Way 1's other half: pull the object out of the reply. Models often wrap JSON in a code fence
    even when asked not to, so strip one if it is there, then let Pydantic do the checking."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return ItineraryReview.model_validate_json((fenced.group(1) if fenced else text).strip())


def print_review(label: str, review: ItineraryReview) -> None:
    print(f"--- {label}: {type(review).__name__} ---")
    print(json.dumps(review.model_dump(), indent=2))
    print()


async def get_itinerary(planner: str | None, model_id: str, profile: str, region: str) -> tuple[str, str]:
    """The itinerary to review, and a label saying where it came from."""
    if planner is None:
        return SAMPLE_ITINERARY, "built-in sample; use --planner for a fresh one"
    name = f"{planner.title()} Planner"
    print(f"Generating an itinerary with {name}...\n")
    # A plain agent with no schema of any kind: its reply is just text
    run = await run_agent("Write today's plan.", make_options(PLANNERS[planner], model_id, profile, region))
    delete_session(run.result.session_id)
    return run.result.result, f"generated by {name}"


async def main(planner: str | None):

    # 1. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    model_id = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", "global.anthropic.claude-sonnet-5")

    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print(f"Claude Model ID:    {model_id}")
    print("--------------------------\n")

    # 2. The schema both ways are built from
    schema = ItineraryReview.model_json_schema()
    print("--- Wanted output (ItineraryReview.model_json_schema()) ---")
    for field, field_schema in schema["properties"].items():
        kind = " | ".join(field_schema["enum"]) if "enum" in field_schema else field_schema["type"]
        print(f"* {field} ({kind}): {field_schema['description']}")
    print()

    print("--- Two ways to get it, since ClaudeAgentOptions has no output_type ---")
    print("1. Ask for JSON: put the schema in the system prompt, then parse the reply yourself.")
    print(f"2. Use a tool as the schema: the agent calls {submit_review.name} and the SDK validates")
    print("   the arguments, so a wrong shape goes back to the agent instead of reaching your code.\n")

    # 3. Get the plan to review: the fixed sample, or a fresh plan written by a planner agent
    itinerary, source = await get_itinerary(planner, model_id, profile, region)
    print(f"--- Itinerary to review ({source}) ---")
    print(f"{itinerary}\n")

    # 4. Way 1: ask for JSON in the system prompt. Nothing enforces it, so the reply is text that
    #    might not parse, and json.dumps of the schema is up to you to write.
    json_instructions = (
        f"{REVIEWER_ROLE}\nReply with a single JSON object matching this schema and nothing else, "
        f"with no code fence and no explanation:\n{json.dumps(schema, indent=2)}"
    )
    print(f"=== Way 1: asking for JSON on {model_id} via Amazon Bedrock ===")
    json_run = await run_agent(itinerary, make_options(json_instructions, model_id, profile, region))
    print(f"Raw reply ({len(json_run.result.result)} chars):")
    print(f"{json_run.result.result}\n")
    try:
        print_review("Parsed with Pydantic", parse_json_reply(json_run.result.result))
    except (ValidationError, ValueError) as error:  # ValueError covers invalid JSON
        print(f"Could not parse the reply: {type(error).__name__}\n{error}\n")

    # 5. Way 2: the tool is the schema. The agent has to fill in its arguments to answer at all.
    print(f"=== Way 2: {submit_review.name} as the schema ===")
    tool_run = await run_agent(
        itinerary,
        make_options(REVIEWER_ROLE, model_id, profile, region, tools=[submit_review], max_turns=3),
    )
    submissions = [call for call in tool_run.calls if call.name.endswith(submit_review.name)]
    if not submissions:
        sys.exit(f"The agent never called {submit_review.name}. Reply was: {tool_run.result.result}")
    # The arguments are the review. Validating again turns the dict into a typed object, and would
    # catch anything the JSON Schema allows but Pydantic doesn't. The last call wins, in case the
    # agent's first attempt was rejected and it tried again.
    review = ItineraryReview.model_validate(submissions[-1].input)
    print(f"{submissions[-1].name} was called with {len(submissions[-1].input)} arguments")
    print_review("Validated tool arguments", review)

    # 6. Because the fields are typed, plain code can act on them without parsing text
    print("--- Using the fields in code ---")
    print(f"review.is_realistic = {review.is_realistic}")
    print(f"review.crowd_risk   = {review.crowd_risk!r}")
    print(f"review.nature_score = {review.nature_score}")
    print(f"len(review.issues)  = {len(review.issues)}\n")

    if review.is_realistic and review.crowd_risk == "low" and review.nature_score >= 7:
        print("Verdict: approve the plan as is.")
    else:
        print(f"Verdict: send back for changes. Top fix: {review.suggested_fix}")

    runs = [json_run, tool_run]
    print("\n--- Run Summary ---")
    print(f"Agent runs: {len(runs)} (one per way)")
    print(f"Cost:       ${sum(run.result.total_cost_usd or 0 for run in runs):.4f}")

    # 7. Clean up: neither run is ever resumed, so remove both saved transcripts
    for run in runs:
        delete_session(run.result.session_id)
    print(f"\n--- {len(runs)} sessions deleted (delete_session) ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review a Tokyo itinerary with structured output.")
    parser.add_argument("--planner", choices=PLANNERS,
                        help="have a planner agent write a fresh itinerary instead of using the built-in bad sample")
    args = parser.parse_args()
    # Model output contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(args.planner))
