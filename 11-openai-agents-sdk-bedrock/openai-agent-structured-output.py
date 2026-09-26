import asyncio
import json
import os
from typing import Literal
from openai import AsyncOpenAI
from openai.providers import bedrock
from pydantic import BaseModel, Field
# Import the explicit tracing control from the agents SDK
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled


# --- Structured output: the agent must reply with an object matching this schema ---
# Field descriptions are sent to the model, so they double as instructions for each field.

class ItineraryReview(BaseModel):
    is_realistic: bool = Field(description="Whether the plan can be done at a relaxed pace in one day")
    crowd_risk: Literal["low", "medium", "high"] = Field(description="How crowded the chosen places and times are likely to be")
    nature_score: int = Field(description="How nature-focused the plan is, from 1 (not at all) to 10 (entirely)")
    issues: list[str] = Field(description="Specific problems with the plan, one short sentence each")
    suggested_fix: str = Field(description="One concrete change that would improve the plan the most")


async def main():

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

    # 5. output_type makes the SDK request JSON matching the schema and parse it into an ItineraryReview
    reviewer = Agent(
        name="Itinerary Reviewer",
        instructions="You review Tokyo day plans for travelers who want a relaxed, nature-focused day away from crowds.",
        model=model_id,
        output_type=ItineraryReview,
    )

    print("--- Agent ---")
    print(f"Name:         {reviewer.name}")
    print(f"Instructions: {reviewer.instructions}")
    print(f"Output type:  {ItineraryReview.__name__}\n")

    print("--- Output schema (ItineraryReview.model_json_schema()) ---")
    for field, schema in ItineraryReview.model_json_schema()["properties"].items():
        kind = " | ".join(schema["enum"]) if "enum" in schema else schema["type"]
        print(f"* {field} ({kind}): {schema['description']}")
    print()

    # A deliberately bad plan: rushed, crowded, and barely any nature
    itinerary = """
Saturday in Tokyo, relaxed nature day:
- 8:00 Shibuya Crossing and Hachiko for photos
- 9:30 Senso-ji and Nakamise Street
- 11:00 Day trip to Mount Fuji 5th Station
- 15:00 Tokyo Disneyland
- 20:00 Robot show in Shinjuku, then karaoke until late
""".strip()
    print("--- Itinerary to review (the prompt) ---")
    print(f"{itinerary}\n")

    print(f"Running agent on {model_id} via Amazon Bedrock...\n")
    result = await Runner.run(reviewer, itinerary)

    # 6. final_output is an ItineraryReview instance, not a string
    review = result.final_output
    print(f"--- Result: {type(review).__name__} ---")
    print(json.dumps(review.model_dump(), indent=2))
    print()

    # 7. Because the fields are typed, plain code can act on them without parsing text
    print("--- Using the fields in code ---")
    print(f"review.is_realistic = {review.is_realistic}")
    print(f"review.crowd_risk   = {review.crowd_risk!r}")
    print(f"review.nature_score = {review.nature_score}")
    print(f"len(review.issues)  = {len(review.issues)}\n")

    if review.is_realistic and review.crowd_risk == "low" and review.nature_score >= 7:
        print("Verdict: approve the plan as is.")
    else:
        print(f"Verdict: send back for changes. Top fix: {review.suggested_fix}")

if __name__ == "__main__":
    asyncio.run(main())
