import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Literal
import boto3
from pydantic import BaseModel, Field
from strands import Agent
from strands.interventions import Deny, Guide, InterventionHandler, Proceed
from strands.models.bedrock import BedrockModel

# Which model family to use, and the env var holding its Bedrock model ID (plus its default).
# All are called through Bedrock's Converse API, so the same BedrockModel class serves each one.
PROVIDERS = {
    "anthropic": ("STRANDS_MODEL_ID_ANTHROPIC", "global.anthropic.claude-sonnet-5"),
    "openai": ("STRANDS_MODEL_ID_OPENAI", "openai.gpt-oss-120b-1:0"),  # OpenAI's open-weight gpt-oss
    "amazon": ("STRANDS_MODEL_ID_AMAZON", "global.amazon.nova-2-lite-v1:0"),  # Amazon's own Nova
}

DEFAULT_REQUEST = "Plan my Saturday in Tokyo in April. I want a relaxed day in nature."


# --- Structured outputs used by the two guardrails' checker agents ---

class TopicCheck(BaseModel):
    is_tokyo_travel: bool = Field(description="Whether the request asks for help planning time in or around Tokyo")
    reason: str = Field(description="One short sentence explaining the decision")


class ItineraryReview(BaseModel):
    is_realistic: bool = Field(description="Whether the plan can be done at a relaxed pace in one day")
    crowd_risk: Literal["low", "medium", "high"] = Field(description="How crowded the chosen places and times are likely to be")
    nature_score: int = Field(description="How nature-focused the plan is, from 1 (not at all) to 10 (entirely)")
    issues: list[str] = Field(description="Specific problems with the plan, one short sentence each")

    def is_problem(self) -> bool:
        return not self.is_realistic or self.crowd_risk == "high" or self.nature_score < 6

    def summary(self) -> str:
        return f"realistic={self.is_realistic}, crowd_risk={self.crowd_risk!r}, nature_score={self.nature_score}"


# --- Guardrails: InterventionHandler subclasses. Strands calls the lifecycle methods they override. ---
# Each one keeps the decisions it made, so the script can print them afterwards.

@dataclass
class TopicGuardrail(InterventionHandler):
    """Input guardrail: before the planner's model is ever called, check the request is about Tokyo."""
    make_checker: callable
    name: str = "tokyo_travel_only"
    checks: list[TopicCheck] = field(default_factory=list)

    async def before_invocation(self, event, **kwargs):
        request = " ".join(block.get("text", "") for message in event.messages for block in message["content"])
        result = await self.make_checker().invoke_async(request, structured_output_model=TopicCheck)
        check = result.structured_output
        self.checks.append(check)
        # Deny cancels the whole call: the planner's model never runs and the reply is "DENIED: <reason>"
        return Proceed() if check.is_tokyo_travel else Deny(reason=check.reason)

    @property
    def blocked(self) -> bool:
        return any(not check.is_tokyo_travel for check in self.checks)


@dataclass
class ItineraryGuardrail(InterventionHandler):
    """Output guardrail: review each finished reply. Send the first bad plan back with the review
    as feedback, and mark the plan blocked if the second one fails too."""
    make_reviewer: callable
    name: str = "itinerary_quality"
    max_revisions: int = 1
    reviews: list[ItineraryReview] = field(default_factory=list)

    async def after_model_call(self, event, **kwargs):
        # This runs after every model call. Only a finished reply (end_turn) is a plan to review.
        response = event.stop_response
        if response is None or response.stop_reason != "end_turn":
            return Proceed()
        plan = "".join(block.get("text", "") for block in response.message["content"])
        result = await self.make_reviewer().invoke_async(plan, structured_output_model=ItineraryReview)
        review = result.structured_output
        self.reviews.append(review)
        if review.is_problem() and len(self.reviews) <= self.max_revisions:
            # Guide discards this reply and calls the model again, with the feedback added as a user message
            return Guide(feedback="Revise the plan. A reviewer found these problems: " + " ".join(review.issues))
        # Guide is the only blocking action here: Deny is ignored after a model call.
        # So a second failure lets the reply through, and the script withholds it (see `blocked`).
        return Proceed()

    @property
    def blocked(self) -> bool:
        return bool(self.reviews) and self.reviews[-1].is_problem()


PLAN_FORMAT = "Reply with a title line and 5 or 6 timed bullet points only."

# Planner name -> system prompt. The good planners should pass the output guardrail; chaotic should trip it.
PLANNERS = {
    "thoughtful": (
        "You write genuinely relaxed, nature-focused Tokyo day plans: gardens, parks and easy hikes, "
        f"few stops close together, quiet times of day. {PLAN_FORMAT}"
    ),
    "offbeat": (
        "You write relaxed, nature-focused Tokyo day plans that go off the beaten path: lesser-known gardens, "
        f"neighborhood shrines, river walks and green spaces most tourists skip. {PLAN_FORMAT}"
    ),
    "chaotic": (
        "You write Tokyo day plans that are overpacked, visit famous crowded spots at peak times, "
        f"and contain almost no nature, while calling it a relaxed day. {PLAN_FORMAT}"
    ),
}


def run_planner(name: str, model: BedrockModel, request: str) -> None:
    """Run one planner behind fresh guardrails and show what each one decided."""

    def make_agent(agent_name: str, system_prompt: str, **kwargs) -> Agent:
        return Agent(name=agent_name, model=model, system_prompt=system_prompt, callback_handler=None, **kwargs)

    # Checkers start fresh for each check, so one check's conversation never leaks into the next
    topic_guard = TopicGuardrail(lambda: make_agent(
        "Topic Checker", "You decide whether a request is about planning travel or leisure time in or around Tokyo."))
    plan_guard = ItineraryGuardrail(lambda: make_agent(
        "Itinerary Reviewer",
        "You review Tokyo day plans for travelers who want a relaxed, nature-focused day away from crowds."))
    # interventions=[...] run in order, so the cheap topic check comes first
    planner = make_agent(f"{name.title()} Planner", PLANNERS[name], interventions=[topic_guard, plan_guard])

    print(f"=== {planner.name} ===")
    start = time.perf_counter()
    result = planner(request)
    duration = time.perf_counter() - start

    if topic_guard.blocked:
        # The planner's model never ran: no plan was generated and no planner tokens were spent
        print(f"Input guardrail TRIPPED: request rejected before planning.\n   {topic_guard.checks[-1].reason}")
        print(f"   Planner reply: {str(result).strip()!r}")
    else:
        print("Input guardrail passed.")
        for attempt, review in enumerate(plan_guard.reviews, 1):
            verdict = "FAILED" if review.is_problem() else "passed"
            print(f"Output guardrail, plan {attempt}: {verdict} ({review.summary()})")
            for issue in review.issues:
                print(f"   - {issue}")
            if review.is_problem() and attempt <= plan_guard.max_revisions:
                print("   -> sent back to the planner with these issues as feedback (Guide)")
        if plan_guard.blocked:
            print(f"\nPlan BLOCKED after {len(plan_guard.reviews)} reviews. Blocked output (shown only for learning):")
        else:
            print("\nPlan delivered:")
        print(str(result).strip())

    usage = planner.event_loop_metrics.accumulated_usage
    print(f"\n({duration:.1f}s; planner tokens {usage['inputTokens']} in, {usage['outputTokens']} out, "
          "checkers not included)\n")


def main(planner_names: list[str], request: str):

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

    print("--- Setup ---")
    print(f"Input guardrail  '{TopicGuardrail.name}' (before_invocation): runs a Topic Checker on the request")
    print("  and returns Deny for anything that isn't about Tokyo travel, so the planner's model never runs.")
    print(f"Output guardrail '{ItineraryGuardrail.name}' (after_model_call): runs an Itinerary Reviewer on each")
    print("  finished plan. A plan that is unrealistic, has high crowd risk or scores below 6 for nature")
    print("  is sent back once with Guide(feedback); if the revision fails too, the plan is withheld.")
    print(f"Protected agents: {', '.join(f'{name.title()} Planner' for name in planner_names)}\n")

    print("--- Request (sent to each planner) ---")
    print(f"{request}\n")

    print(f"Running on {model_id} via Amazon Bedrock...\n")
    for name in planner_names:
        run_planner(name, model, request)

if __name__ == "__main__":
    # Model output contains characters like en dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Show input and output guardrails blocking or passing Tokyo day plans.")
    parser.add_argument("--planner", choices=PLANNERS,
                        help="run only this planner (default: run every planner)")
    parser.add_argument("--request", default=DEFAULT_REQUEST,
                        help="the traveler's request; try an off-topic one to trip the input guardrail")
    args = parser.parse_args()
    main([args.planner] if args.planner else list(PLANNERS), args.request)
