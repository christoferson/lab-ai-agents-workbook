import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Annotated, Literal
from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, HookMatcher, ResultMessage, ToolUseBlock, create_sdk_mcp_server,
    delete_session, query, tool,
)
from pydantic import BaseModel, Field

SERVER_NAME = "guarded"
HOOK_TIMEOUT = 180  # seconds; a guardrail here runs a whole agent, which takes longer than the 60s default
DEFAULT_REQUEST = "Plan my Saturday in Tokyo in April. I want a relaxed day in nature."


# --- What the two guardrails decide with. As in the structured output example, each model doubles as
#     a tool schema, so the checker agents answer with a filled-in form instead of text. ---

class TopicCheck(BaseModel):
    is_tokyo_travel: bool = Field(description="Whether the request asks for help planning time in or around Tokyo")
    reason: str = Field(description="One short sentence explaining the decision")


class ItineraryReview(BaseModel):
    is_realistic: bool = Field(description="Whether the plan can be done at a relaxed pace in one day")
    crowd_risk: Literal["low", "medium", "high"] = Field(description="How crowded the chosen places and times are likely to be")
    nature_score: int = Field(description="How nature-focused the plan is, from 1 (not at all) to 10 (entirely)")
    issues: list[str] = Field(description="Specific problems with the plan, one short sentence each")


def passes_review(review: ItineraryReview | None) -> bool:
    """The output guardrail's bar. A missing review counts as a failure."""
    return bool(review) and review.is_realistic and review.crowd_risk != "high" and review.nature_score >= 6


# --- Tools. The planner delivers through submit_plan, which is what makes its output something a
#     hook can inspect; the two checkers answer through a tool built from their schema. ---

@tool("submit_plan", "Submit your finished day plan for the traveler. This is the only way to deliver it.",
      {"plan": Annotated[str, "The full day plan, as a title line and timed bullet points."]})
async def submit_plan(args):
    return {"content": [{"type": "text", "text": "Plan accepted and delivered to the traveler."}]}


@tool("submit_topic_check", "Submit your decision about the request.", TopicCheck.model_json_schema())
async def submit_topic_check(args):
    return {"content": [{"type": "text", "text": "Decision recorded."}]}


@tool("submit_review", "Submit your review of the plan.", ItineraryReview.model_json_schema())
async def submit_review(args):
    return {"content": [{"type": "text", "text": "Review recorded."}]}


TOPIC_CHECKER_ROLE = "You decide whether a request is about planning travel or leisure time in or around Tokyo."
REVIEWER_ROLE = "You review Tokyo day plans for travelers who want a relaxed, nature-focused day away from crowds."


@dataclass
class Run:
    result: ResultMessage
    calls: list[ToolUseBlock]


@dataclass
class Bedrock:
    """Where every run gets its model and credentials. The guardrails start runs of their own, so they
    hold one of these too, and every run is recorded here for the cost summary and the clean-up."""
    model_id: str
    profile: str
    region: str
    runs: list[ResultMessage] = field(default_factory=list)

    def options(self, system_prompt: str, tools: list = (), max_turns: int = 1, hooks: dict | None = None) -> ClaudeAgentOptions:
        tool_options = {}
        if tools:
            tool_options = {
                "mcp_servers": {SERVER_NAME: create_sdk_mcp_server(name=SERVER_NAME, tools=list(tools))},
                "allowed_tools": [tool_name(t) for t in tools],
            }
        return ClaudeAgentOptions(
            model=self.model_id,
            system_prompt=system_prompt,
            tools=[],  # no built-in tools (file access, shell, ...)
            max_turns=max_turns,
            hooks=hooks,
            setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
            env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": self.profile, "AWS_REGION": self.region},
            **tool_options,
        )

    async def run(self, prompt: str, options: ClaudeAgentOptions) -> Run:
        """Run one agent, keeping its final message and any tool calls it made."""
        calls: list[ToolUseBlock] = []
        final = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                calls += [block for block in message.content if isinstance(block, ToolUseBlock)]
            elif isinstance(message, ResultMessage):
                final = message
        self.runs.append(final)
        return Run(final, calls)

    async def ask_for(self, schema: type[BaseModel], schema_tool, system_prompt: str, prompt: str):
        """Run a checker agent that can only answer by calling schema_tool, and return the typed object."""
        run = await self.run(prompt, self.options(system_prompt, tools=[schema_tool], max_turns=3))
        answers = [call for call in run.calls if call.name == tool_name(schema_tool)]
        return schema.model_validate(answers[-1].input) if answers else None


def tool_name(sdk_tool) -> str:
    """The name the agent sees for an MCP tool: mcp__<server>__<tool>."""
    return f"mcp__{SERVER_NAME}__{sdk_tool.name}"


# --- The guardrails. The Claude Agent SDK has no guardrail of its own, so each one is a hook: a
#     callback the CLI runs at a fixed point in the loop, which can block what happens next. ---

@dataclass
class TopicGuardrail:
    """Input guardrail, as a UserPromptSubmit hook: checks the request before the planner is given it."""
    bedrock: Bedrock
    check: TopicCheck | None = None

    async def hook(self, hook_input, tool_use_id, context) -> dict:
        self.check = await self.bedrock.ask_for(TopicCheck, submit_topic_check, TOPIC_CHECKER_ROLE, hook_input["prompt"])
        if self.check and not self.check.is_tokyo_travel:
            # "block" throws the prompt away, so the planner never runs and spends no tokens on it
            return {"decision": "block", "reason": f"Off topic: {self.check.reason}"}
        return {}  # nothing to say: the prompt goes through unchanged


@dataclass
class QualityGuardrail:
    """Output guardrail, as a PreToolUse hook on submit_plan: reviews the plan before it counts as
    delivered. Unlike a tripwire that just raises, a denial goes back to the planner as feedback,
    so it gets one chance to fix the plan; a second failure stops the run."""
    bedrock: Bedrock
    reviews: list[ItineraryReview | None] = field(default_factory=list)
    approved_plan: str | None = None
    stopped: bool = False

    async def hook(self, hook_input, tool_use_id, context) -> dict:
        plan = hook_input["tool_input"]["plan"]
        review = await self.bedrock.ask_for(ItineraryReview, submit_review, REVIEWER_ROLE, plan)
        self.reviews.append(review)

        if passes_review(review):
            self.approved_plan = plan
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}

        issues = " ".join(review.issues) if review else "The plan could not be reviewed."
        if len(self.reviews) == 1:
            # "deny" blocks this call only. The reason is written for the planner, which sees it as the
            # tool's result and can submit a better plan.
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Plan rejected. {issues} Revise the plan and submit it again.",
            }}
        self.stopped = True
        # continue_=False ends the whole run at once, and nothing is delivered
        return {"continue_": False, "stopReason": "Plan rejected twice; the run was stopped."}


PLAN_FORMAT = "Reply with a title line and 5 or 6 timed bullet points only."
SUBMIT_RULE = "Deliver every plan by calling submit_plan. Never write the plan as a normal reply."

# Planner name -> system prompt. The good planners should pass the output guardrail; chaotic should not.
PLANNERS = {
    "thoughtful": (
        "You write genuinely relaxed, nature-focused Tokyo day plans: gardens, parks and easy hikes, "
        f"few stops close together, quiet times of day. {PLAN_FORMAT} {SUBMIT_RULE}"
    ),
    "offbeat": (
        "You write relaxed, nature-focused Tokyo day plans that go off the beaten path: lesser-known gardens, "
        f"neighborhood shrines, river walks and green spaces most tourists skip. {PLAN_FORMAT} {SUBMIT_RULE}"
    ),
    "chaotic": (
        "You write Tokyo day plans that are overpacked, visit famous crowded spots at peak times, "
        f"and contain almost no nature, while calling it a relaxed day. {PLAN_FORMAT} {SUBMIT_RULE}"
    ),
}


async def run_planner(name: str, request: str, bedrock: Bedrock) -> None:
    """Run one planner behind both guardrails and show which one, if any, stopped it."""
    title = f"{name.title()} Planner"
    print(f"=== {title} ===")

    # Fresh guardrails per run: each keeps the decisions it made about this planner
    topic = TopicGuardrail(bedrock)
    quality = QualityGuardrail(bedrock)
    options = bedrock.options(
        PLANNERS[name],
        tools=[submit_plan],
        max_turns=8,  # room for a first plan, the revision the guardrail asks for, and the replies
        hooks={
            "UserPromptSubmit": [HookMatcher(hooks=[topic.hook], timeout=HOOK_TIMEOUT)],
            # matcher picks the tool this hook guards, by the name the agent sees
            "PreToolUse": [HookMatcher(matcher=tool_name(submit_plan), hooks=[quality.hook], timeout=HOOK_TIMEOUT)],
        },
    )
    run = await bedrock.run(request, options)

    if topic.check:
        verdict = "on topic" if topic.check.is_tokyo_travel else "BLOCKED"
        print(f"[input guardrail] Topic Checker: {verdict} - {topic.check.reason}")
    if topic.check and not topic.check.is_tokyo_travel:
        print(f"The planner never ran ({run.result.num_turns} turns), so only the topic check cost anything.\n")
        return

    for attempt, review in enumerate(quality.reviews, 1):
        if review is None:
            print(f"[output guardrail] attempt {attempt}: no review came back -> treated as a failure")
            continue
        outcome = "allowed" if passes_review(review) else "denied"
        print(f"[output guardrail] attempt {attempt}: realistic={review.is_realistic}, "
              f"crowd_risk={review.crowd_risk!r}, nature_score={review.nature_score} -> {outcome}")
        if not passes_review(review):
            for issue in review.issues:
                print(f"   - {issue}")

    if quality.approved_plan:
        print(f"\nPlan delivered:\n{quality.approved_plan}\n")
    elif quality.stopped:
        print("\nNothing delivered: the guardrail stopped the run after the second bad plan.\n")
    else:
        print(f"\nNothing delivered: the planner never called {submit_plan.name}. It said:\n{run.result.result}\n")


async def main(planner_names: list[str], request: str):

    # 1. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    model_id = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", "global.anthropic.claude-sonnet-5")

    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print(f"Claude Model ID:    {model_id}")
    print("--------------------------\n")

    bedrock = Bedrock(model_id, profile, region)

    print("--- Setup: two guardrails, both hooks ---")
    print("Input guardrail (UserPromptSubmit hook): runs Topic Checker on the request before the planner")
    print('  is given it, and blocks anything that is not about Tokyo travel with decision="block".')
    print(f"Output guardrail (PreToolUse hook on {submit_plan.name}): runs Itinerary Reviewer on each")
    print("  submitted plan, and denies one that is unrealistic, crowded or scores below 6 for nature.")
    print("  The planner reads the denial and gets one chance to revise; a second failure stops the run.")
    print(f"Protected agents: {', '.join(f'{name.title()} Planner' for name in planner_names)}\n")

    print("--- Request (sent to each planner) ---")
    print(f"{request}\n")

    print(f"Running on {model_id} via Amazon Bedrock...\n")
    for name in planner_names:
        await run_planner(name, request, bedrock)

    # 2. Every run is in bedrock.runs: the planners plus each checker the guardrails started
    print("--- Run Summary ---")
    print(f"Agent runs: {len(bedrock.runs)} (planners, topic checks and plan reviews)")
    print(f"Cost:       ${sum(run.total_cost_usd or 0 for run in bedrock.runs):.4f}")

    # 3. Clean up: none of these sessions is ever resumed, so remove their transcripts
    sessions = {run.session_id for run in bedrock.runs}
    for session_id in sessions:
        delete_session(session_id)
    print(f"\n--- {len(sessions)} sessions deleted (delete_session) ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show input and output guardrails blocking or passing Tokyo day plans.")
    parser.add_argument("--planner", choices=PLANNERS,
                        help="run only this planner (default: run every planner)")
    parser.add_argument("--request", default=DEFAULT_REQUEST,
                        help="the traveler's request; try an off-topic one to trip the input guardrail")
    args = parser.parse_args()
    # Model output contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main([args.planner] if args.planner else list(PLANNERS), args.request))
