import argparse
import asyncio
import os
from typing import Literal
from openai import AsyncOpenAI
from openai.providers import bedrock
from pydantic import BaseModel, Field
# Import the explicit tracing control from the agents SDK
from agents import (
    Agent, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, Runner,
    input_guardrail, output_guardrail, set_default_openai_client, set_tracing_disabled,
)

DEFAULT_REQUEST = "Plan my Saturday in Tokyo in April. I want a relaxed day in nature."


# --- Structured outputs used by the two guardrails ---

class TopicCheck(BaseModel):
    is_tokyo_travel: bool = Field(description="Whether the request asks for help planning time in or around Tokyo")
    reason: str = Field(description="One short sentence explaining the decision")


class ItineraryReview(BaseModel):
    is_realistic: bool = Field(description="Whether the plan can be done at a relaxed pace in one day")
    crowd_risk: Literal["low", "medium", "high"] = Field(description="How crowded the chosen places and times are likely to be")
    nature_score: int = Field(description="How nature-focused the plan is, from 1 (not at all) to 10 (entirely)")
    issues: list[str] = Field(description="Specific problems with the plan, one short sentence each")


def build_topic_guardrail(checker: Agent):
    """Create an input guardrail that runs the checker on the user's request."""

    # run_in_parallel=False: finish the check before the planner starts, so an
    # off-topic request never spends planner tokens (at the cost of some latency)
    @input_guardrail(name="tokyo_travel_only", run_in_parallel=False)
    async def topic_guardrail(ctx, agent, input):
        result = await Runner.run(checker, input, context=ctx.context)
        check = result.final_output
        # tripwire_triggered=True makes Runner.run raise InputGuardrailTripwireTriggered
        return GuardrailFunctionOutput(output_info={"check": check}, tripwire_triggered=not check.is_tokyo_travel)

    return topic_guardrail


def build_itinerary_guardrail(reviewer: Agent):
    """Create an output guardrail that runs the reviewer on the planner's output."""

    @output_guardrail(name="itinerary_quality")
    async def itinerary_guardrail(ctx, agent, output):
        result = await Runner.run(reviewer, output, context=ctx.context)
        review = result.final_output
        is_problem = not review.is_realistic or review.crowd_risk == "high" or review.nature_score < 6
        # tripwire_triggered=True makes Runner.run raise OutputGuardrailTripwireTriggered
        return GuardrailFunctionOutput(output_info={"review": review}, tripwire_triggered=is_problem)

    return itinerary_guardrail


PLAN_FORMAT = "Reply with a title line and 5 or 6 timed bullet points only."

# Planner name -> instructions. The good planners should pass the output guardrail; chaotic should trip it.
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


def create_planner(name: str, model_id: str, input_guard, output_guard) -> Agent:
    """Factory: return the planner agent for the given name, protected by both guardrails."""
    return Agent(
        name=f"{name.title()} Planner",
        instructions=PLANNERS[name],
        model=model_id,
        input_guardrails=[input_guard],
        output_guardrails=[output_guard],
    )


def print_review(review: ItineraryReview):
    print(f"   realistic={review.is_realistic}, crowd_risk={review.crowd_risk!r}, nature_score={review.nature_score}")
    for issue in review.issues:
        print(f"   - {issue}")


async def run_planner(planner: Agent, request: str):
    """Run one planner and show which guardrail, if any, stopped it."""
    print(f"=== {planner.name} ===")
    try:
        result = await Runner.run(planner, request)
    except InputGuardrailTripwireTriggered as e:
        # The planner never ran: no plan was generated and no planner tokens were spent
        check = e.guardrail_result.output.output_info["check"]
        print(f"Input guardrail TRIPPED: request rejected before planning.\n   {check.reason}\n")
        return
    except OutputGuardrailTripwireTriggered as e:
        # The planner ran, but its plan is stopped before it reaches the traveler
        print("Input guardrail passed.")
        print("Output guardrail TRIPPED: plan blocked.")
        print_review(e.guardrail_result.output.output_info["review"])
        print(f"\nBlocked output (shown only for learning):\n{e.guardrail_result.agent_output}\n")
        return

    print("Input guardrail passed.")
    print("Output guardrail passed: plan delivered.")
    print_review(result.output_guardrail_results[0].output.output_info["review"])
    print(f"\nDelivered plan:\n{result.final_output}\n")


async def main(planner_names: list[str], request: str):

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

    # 5. The checker agents and the guardrails built on them
    topic_checker = Agent(
        name="Topic Checker",
        instructions="You decide whether a request is about planning travel or leisure time in or around Tokyo.",
        model=model_id,
        output_type=TopicCheck,
    )
    reviewer = Agent(
        name="Itinerary Reviewer",
        instructions="You review Tokyo day plans for travelers who want a relaxed, nature-focused day away from crowds.",
        model=model_id,
        output_type=ItineraryReview,
    )
    input_guard = build_topic_guardrail(topic_checker)
    output_guard = build_itinerary_guardrail(reviewer)

    # 6. Build the selected planners, all protected by the same two guardrails
    planners = [create_planner(name, model_id, input_guard, output_guard) for name in planner_names]

    print("--- Setup ---")
    print(f"Input guardrail  '{input_guard.get_name()}': runs {topic_checker.name} on the request before")
    print("  the planner starts, and rejects anything that isn't about Tokyo travel.")
    print(f"Output guardrail '{output_guard.get_name()}': runs {reviewer.name} on each plan and trips if it is")
    print("  unrealistic, has high crowd risk, or scores below 6 for nature.")
    print(f"Protected agents: {', '.join(p.name for p in planners)}\n")

    print("--- Request (sent to each planner) ---")
    print(f"{request}\n")

    print(f"Running on {model_id} via Amazon Bedrock...\n")
    for planner in planners:
        await run_planner(planner, request)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show input and output guardrails blocking or passing Tokyo day plans.")
    parser.add_argument("--planner", choices=PLANNERS,
                        help="run only this planner (default: run every planner)")
    parser.add_argument("--request", default=DEFAULT_REQUEST,
                        help="the traveler's request; try an off-topic one to trip the input guardrail")
    args = parser.parse_args()
    asyncio.run(main([args.planner] if args.planner else list(PLANNERS), args.request))
