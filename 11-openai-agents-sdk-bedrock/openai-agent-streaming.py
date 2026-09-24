import asyncio
import os
from openai import AsyncOpenAI
from openai.providers import bedrock
from openai.types.responses import ResponseTextDeltaEvent
# Import the explicit tracing control from the agents SDK
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled


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
        name="Agent Tutor",
        instructions="You explain AI agent concepts clearly and concisely to developers.",
        model=model_id
    )

    print("--- Agent ---")
    print(f"Name:         {agent.name}")
    print(f"Instructions: {agent.instructions}\n")

    # 6. Run the agentic loop in streaming mode
    prompt = "List the 5 core building blocks of an AI agent, one line each."
    print("--- Prompt ---")
    print(f"{prompt}\n")

    print(f"Streaming response from {model_id} via Amazon Bedrock...\n")
    result = Runner.run_streamed(agent, input=prompt)

    # Print each text delta as it arrives
    print("--- Agent Response (streaming) ---")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    print()

if __name__ == "__main__":
    asyncio.run(main())
