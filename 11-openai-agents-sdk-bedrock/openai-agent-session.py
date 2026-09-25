import asyncio
import os
from pathlib import Path
from openai import AsyncOpenAI
from openai.providers import bedrock
# Import the explicit tracing control from the agents SDK
from agents import Agent, Runner, SQLiteSession, set_default_openai_client, set_tracing_disabled

# On-disk session store next to this script. Use SQLiteSession("id") with no path for in-memory only.
DB_PATH = Path(__file__).with_name("sessions.db")
SESSION_ID = "tokyo-trip"


def print_history(items):
    """Print each conversation item as 'role: text' so the history is easy to read."""
    for i, item in enumerate(items, 1):
        content = item.get("content")
        if isinstance(content, list):  # assistant messages hold a list of content parts
            content = "".join(part.get("text", "") for part in content)
        elif content is None:  # e.g. reasoning items carry no visible text
            content = "(no text content)"
        print(f"  {i}. {item.get('role', item.get('type'))}: {content}")


async def ask(agent, session, prompt):
    """Run one turn; the session loads earlier turns before the call and saves the new ones after."""
    print(f"User:  {prompt}")
    response = await Runner.run(agent, prompt, session=session)
    print(f"Agent: {response.final_output}\n")


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

    # 6. Open the session. Unlike to_input_list(), you never pass history yourself:
    #    Runner.run(..., session=session) reads and writes it in SQLite automatically.
    session = SQLiteSession(SESSION_ID, DB_PATH)
    await session.clear_session()  # start fresh so every run of this demo is the same
    print(f"--- Session '{SESSION_ID}' stored in {DB_PATH.name} ---\n")

    print("=== Turn 1 ===")
    await ask(agent, session, "I'm spending three relaxed days in Tokyo in April. "
                              "I love nature, like gardens, parks and easy hikes, and I'd rather avoid crowds.")

    print("=== Turn 2 (same session, no history passed) ===")
    await ask(agent, session, "What should I do on my first morning?")

    print("--- What SQLite has stored (session.get_items()) ---")
    print_history(await session.get_items())
    print()

    # 7. Simulate an app restart: close the session, then reopen it by the same ID and file
    session.close()
    session = SQLiteSession(SESSION_ID, DB_PATH)

    print("=== Turn 3 (after reopening the session from disk) ===")
    await ask(agent, session, "Suggest an easy day trip out of the city for day three.")

    session.close()

if __name__ == "__main__":
    asyncio.run(main())
