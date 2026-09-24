import asyncio
import os
from openai import AsyncOpenAI
from openai.providers import bedrock  
# Import the explicit tracing control from the agents SDK
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled

async def main():

    # 1. Disable the telemetry trace exporter directly via the SDK function
    set_tracing_disabled(True)

    # 2. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    
    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print("--------------------------\n")

    # 3. Instantiate Bedrock client using the official provider framework.
    bedrock_client = AsyncOpenAI(
        provider=bedrock(
            region=region
        )
    )

    # 4. Register your Bedrock client as default for the OpenAI Agents SDK
    set_default_openai_client(bedrock_client)

    # 5. Initialize your Agent with the exact Bedrock model ID for GPT-5.6 Luna
    agent = Agent(
        name="Jokester", 
        instructions="You are a quick and witty joke teller.", 
        model="openai.gpt-5.6-luna"  
    )

    # 6. Run the agentic loop asynchronously
    print("Sending prompt to GPT-5.6 Luna via Amazon Bedrock...")
    result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")

    # Output the final result string
    print("\n--- Agent Response ---")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
