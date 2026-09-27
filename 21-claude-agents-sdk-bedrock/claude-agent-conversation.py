import asyncio
import os
import sys
from claude_agent_sdk import (
    ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, delete_session, get_session_messages,
)


async def ask(client: ClaudeSDKClient, prompt: str) -> ResultMessage:
    """Send one message on a client and return the ResultMessage (final text and session ID)."""
    await client.query(prompt)
    async for message in client.receive_response():
        if isinstance(message, ResultMessage):
            return message


def print_history(session_id: str):
    """Print each message the CLI saved for a session as 'role: text' so the history is easy to read."""
    i = 0
    for message in get_session_messages(session_id):
        content = message.message.get("content")
        if isinstance(content, list):  # assistant messages hold a list of content blocks
            content = "".join(block.get("text", "") for block in content)
        if not content:  # e.g. a message holding only the model's thinking, with no visible text
            continue
        i += 1
        print(f"  {i}. {message.type}: {content}")


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

    # 2. Configure the agent
    system_prompt = (
        "You are a travel planner. Tailor suggestions to the traveler's destination and needs. "
        "If you are missing details, ask for them. Keep replies to one or two sentences."
    )
    options = ClaudeAgentOptions(
        model=model_id,
        system_prompt=system_prompt,
        tools=[],  # no built-in tools (file access, shell, ...): a plain question-and-answer agent
        max_turns=1,
        setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
        env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region},
    )

    print("--- Agent ---")
    print(f"System prompt: {system_prompt}\n")

    # How memory works here: you never pass the history or a session ID. Each ClaudeSDKClient runs its
    # own Claude Code CLI process, and that process remembers everything said on it (one session).
    # So "with history" means "ask the same client again", and "without history" means "use another client":
    # client_with_conversation is asked both turns, client_no_conversation only the follow-up question.
    first_prompt = (
        "I'm spending three relaxed days in Tokyo in April. "
        "I love nature, like gardens, parks and easy hikes, and I'd rather avoid crowds."
    )
    follow_up = "What should I do on my first morning?"

    async with ClaudeSDKClient(options=options) as client_with_conversation:

        # 3. Turn 1 on `client_with_conversation`: share the trip details
        print("=== Turn 1: ask(client_with_conversation, ...) ===")
        print(f"User:  {first_prompt}")
        turn1 = await ask(client_with_conversation, first_prompt)
        print(f"Agent: {turn1.result}")
        print(f"Session: {turn1.session_id}\n")

        # 4. Turn 2 WITHOUT history: the same question on a second client, which starts an empty session
        async with ClaudeSDKClient(options=options) as client_no_conversation:
            print("=== Turn 2, WITHOUT history: ask(client_no_conversation, ...) ===")
            print(f"User:  {follow_up}")
            forgetful = await ask(client_no_conversation, follow_up)
            print(f"Agent: {forgetful.result}")
            print(f"Session: {forgetful.session_id} (different from turn 1, so it knows nothing)\n")

        # 5. Turn 2 WITH history: the same question on the client from turn 1, which still holds that turn
        print("=== Turn 2, WITH history: ask(client_with_conversation, ...) ===")
        print(f"User:  {follow_up}")
        turn2 = await ask(client_with_conversation, follow_up)
        print(f"Agent: {turn2.result}")
        same = "same as turn 1" if turn2.session_id == turn1.session_id else "different from turn 1"
        print(f"Session: {turn2.session_id} ({same})\n")

    # 6. The CLI also saves each session as a transcript file; get_session_messages() reads it back
    print(f"--- History of client_with_conversation (session {turn1.session_id}) ---")
    print_history(turn1.session_id)
    print(f"\n--- History of client_no_conversation (session {forgetful.session_id}) ---")
    print_history(forgetful.session_id)

    # 7. Clean up: delete_session() removes each saved transcript, so no files are left behind
    for session_id in (turn1.session_id, forgetful.session_id):
        delete_session(session_id)
    print("\n--- Both sessions deleted (delete_session) ---")

if __name__ == "__main__":
    # Model output contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
