import asyncio
import os
from openai import AsyncOpenAI
from openai.providers import bedrock
# Import the explicit tracing control from the agents SDK
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled


def print_history(items):
    """Print each conversation item as 'role: text' so the history is easy to read."""
    for i, item in enumerate(items, 1):
        content = item.get("content")
        if isinstance(content, list):  # assistant messages hold a list of content parts
            content = "".join(part.get("text", "") for part in content)
        elif content is None:  # e.g. reasoning items carry no visible text
            content = "(no text content)"
        print(f"  {i}. {item.get('role', item.get('type'))}: {content}")


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

    # 5. Initialize your Agent with the Bedrock model ID (BEDROCK_MODEL_ID, default GPT-5.6 Luna)
    agent = Agent(
        name="Travel Planner",
        instructions=(
            "You are a travel planner. Tailor suggestions to the traveler's destination and needs. "
            "If you are missing details, ask for them. Keep replies to one or two sentences."
        ),
        model=model_id
    )

    print("--- Agent ---")
    print(f"Name:         {agent.name}")
    print(f"Instructions: {agent.instructions}\n")

    # Each Runner.run call is stateless: the agent only knows what you pass in as input.

    # 6. Turn 1: share the trip details
    first_prompt = (
        "I'm spending three relaxed days in Tokyo in April. "
        "I love nature, like gardens, parks and easy hikes, and I'd rather avoid crowds."
    )
    print("=== Turn 1 ===")
    print(f"User:  {first_prompt}")
    response = await Runner.run(agent, first_prompt)
    print(f"Agent: {response.final_output}\n")

    # 7. to_input_list() returns the whole conversation so far (your input + the agent's output)
    print("--- History after turn 1 (response.to_input_list()) ---")
    print_history(response.to_input_list())
    print()

    # 8. Turn 2 without history: a fresh run has no memory of turn 1
    follow_up = "What should I do on my first morning?"
    print("=== Turn 2, WITHOUT history ===")
    print(f"User:  {follow_up}")
    forgetful = await Runner.run(agent, follow_up)
    print(f"Agent: {forgetful.final_output}\n")

    # 9. Turn 2 with history: append the new message to the previous conversation
    next_input = response.to_input_list() + [{"role": "user", "content": follow_up}]
    print("=== Turn 2, WITH history ===")
    print("Input sent to the agent:")
    print_history(next_input)
    response = await Runner.run(agent, next_input)
    print(f"Agent: {response.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
