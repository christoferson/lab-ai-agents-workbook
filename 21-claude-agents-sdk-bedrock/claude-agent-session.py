import asyncio
import os
import sys
import uuid
from pathlib import Path
from claude_agent_sdk import (
    ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, delete_session, get_session_messages,
)

# Session IDs must be UUIDs, so derive a fixed one from a readable name. The same name always gives
# the same ID, which is how a later run (or a restarted app) finds this conversation again.
SESSION_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "tokyo-trip"))

SYSTEM_PROMPT = (
    "You are a travel planner. Tailor suggestions to the traveler's destination and needs. "
    "If you are missing details, ask for them. Keep replies to one or two sentences."
)


def make_options(model_id: str, profile: str, region: str, **session) -> ClaudeAgentOptions:
    """Factory: the agent's options, plus how to pick its session (session_id=... or resume=...)."""
    return ClaudeAgentOptions(
        model=model_id,
        system_prompt=SYSTEM_PROMPT,
        tools=[],  # no built-in tools (file access, shell, ...): a plain question-and-answer agent
        max_turns=1,
        setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
        env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region},
        **session,
    )


def delete_saved_session():
    """Delete the session's saved transcript, if there is one."""
    try:
        delete_session(SESSION_ID)
    except FileNotFoundError:
        pass


def transcript_path() -> str:
    """The CLI saves each session as ~/.claude/projects/<project folder>/<session id>.jsonl."""
    path = next(Path.home().glob(f".claude/projects/*/{SESSION_ID}.jsonl"), None)
    # Show it relative to the home folder, which keeps your user name out of shared output
    return f"~/{path.relative_to(Path.home()).as_posix()}" if path else "(not found)"


def print_history(session_id: str):
    """Print each saved message as 'role: text', skipping ones with no visible text (e.g. only thinking)."""
    i = 0
    for message in get_session_messages(session_id):
        content = message.message.get("content")
        if isinstance(content, list):  # assistant messages hold a list of content blocks
            content = "".join(block.get("text", "") for block in content)
        if not content:
            continue
        i += 1
        print(f"  {i}. {message.type}: {content}")


async def ask(client: ClaudeSDKClient, prompt: str) -> ResultMessage:
    """Run one turn on a client and print it, with the session it ran in."""
    print(f"User:    {prompt}")
    await client.query(prompt)
    async for message in client.receive_response():
        if isinstance(message, ResultMessage):
            print(f"Agent:   {message.result}")
            print(f"Session: {message.session_id}\n")
            return message


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

    print("--- Agent ---")
    print(f"System prompt: {SYSTEM_PROMPT}\n")

    # 2. Start the session under our own ID. In claude-agent-conversation.py the client held the
    #    conversation only while it was open; here the ID lets a later client bring it back.
    delete_saved_session()  # in case an earlier run stopped before its clean-up
    print(f"--- Session 'tokyo-trip' = {SESSION_ID} ---\n")

    async with ClaudeSDKClient(options=make_options(model_id, profile, region, session_id=SESSION_ID)) as client:
        print("=== Turn 1 (ClaudeAgentOptions(session_id=SESSION_ID): start the session) ===")
        await ask(client, "I'm spending three relaxed days in Tokyo in April. "
                          "I love nature, like gardens, parks and easy hikes, and I'd rather avoid crowds.")

        print("=== Turn 2 (same client, so same session) ===")
        await ask(client, "What should I do on my first morning?")

    # 3. The client is closed now, which also ends its CLI process: all that remains is the saved transcript
    print(f"--- What the CLI has stored ({transcript_path()}) ---")
    print_history(SESSION_ID)
    print()

    # 4. Simulate an app restart: a brand-new client, told only the session ID to resume
    async with ClaudeSDKClient(options=make_options(model_id, profile, region, resume=SESSION_ID)) as client:
        print("=== Turn 3 (new client, ClaudeAgentOptions(resume=SESSION_ID): reload the session) ===")
        await ask(client, "Suggest an easy day trip out of the city for day three.")

    # 5. Clean up: delete_session() removes the saved transcript, so the session can't be resumed any more
    saved_path = transcript_path()  # look up the path first: after deleting, there is nothing to find
    delete_saved_session()
    status = "removed" if transcript_path() == "(not found)" else "still there"
    print(f"--- Session deleted (delete_session): {saved_path} {status} ---")

if __name__ == "__main__":
    # Model output contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
