import os
import sys
from pathlib import Path
import boto3
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.session.file_session_manager import FileSessionManager

# Which model family to use, and the env var holding its Bedrock model ID (plus its default).
# All are called through Bedrock's Converse API, so the same BedrockModel class serves each one.
PROVIDERS = {
    "anthropic": ("STRANDS_MODEL_ID_ANTHROPIC", "global.anthropic.claude-sonnet-5"),
    "openai": ("STRANDS_MODEL_ID_OPENAI", "openai.gpt-oss-120b-1:0"),  # OpenAI's open-weight gpt-oss
    "amazon": ("STRANDS_MODEL_ID_AMAZON", "global.amazon.nova-2-lite-v1:0"),  # Amazon's own Nova
}

# On-disk session store next to this script (git-ignored). Without storage_dir it is ~/.strands/sessions.
SESSIONS_DIR = Path(__file__).with_name("sessions")
SESSION_ID = "tokyo-trip"
AGENT_ID = "travel-planner"  # one session can hold several agents, each under its own ID

SYSTEM_PROMPT = (
    "You are a travel planner. Tailor suggestions to the traveler's destination and needs. "
    "If you are missing details, ask for them. Keep replies to one or two sentences."
)


def make_agent(model: BedrockModel) -> Agent:
    """Factory: a new Agent attached to the saved session. If the session already has messages for
    AGENT_ID, the session manager loads them into agent.messages right here, before any call."""
    session_manager = FileSessionManager(session_id=SESSION_ID, storage_dir=str(SESSIONS_DIR))
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        agent_id=AGENT_ID,
        session_manager=session_manager,
        callback_handler=None,
    )


def delete_saved_session():
    """Delete the session folder, if there is one. (Creating the manager creates the session, so
    there is always something to delete afterwards.)"""
    FileSessionManager(session_id=SESSION_ID, storage_dir=str(SESSIONS_DIR)).delete_session(SESSION_ID)


def ask(agent: Agent, prompt: str):
    """Run one turn. The session manager saves each new message to disk as it is added."""
    print(f"User:  {prompt}")
    print(f"Agent: {str(agent(prompt)).strip()}\n")


def print_history(agent: Agent):
    """Print agent.messages as 'role: text', showing only text blocks (not gpt-oss reasoning)."""
    for i, message in enumerate(agent.messages, 1):
        text = "".join(block.get("text", "") for block in message["content"]).strip()
        print(f"  {i}. {message['role']}: {text}")


def print_stored_files():
    """List the files the session manager wrote, relative to the repo so no user name is shown."""
    repo = SESSIONS_DIR.parent.parent
    for path in sorted(SESSIONS_DIR.rglob("*.json")):
        print(f"  {path.relative_to(repo).as_posix()}")


def main():

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

    print("--- Agent ---")
    print(f"System prompt: {SYSTEM_PROMPT}\n")

    # 2. In strands-agent-conversation.py the history lived only in agent.messages, so it ended with
    #    the object. A session manager also writes every message to disk, under a session ID.
    delete_saved_session()  # start fresh so every run of this demo is the same
    print(f"--- Session '{SESSION_ID}' stored in {SESSIONS_DIR.name}/ (FileSessionManager) ---\n")

    agent = make_agent(model)
    print("=== Turn 1 ===")
    ask(agent, "I'm spending three relaxed days in Tokyo in April. "
               "I love nature, like gardens, parks and easy hikes, and I'd rather avoid crowds.")

    print("=== Turn 2 (same agent, so same conversation) ===")
    ask(agent, "What should I do on my first morning?")

    # 3. One JSON file per message, plus one for the session and one for the agent
    print("--- What the session manager has stored ---")
    print_stored_files()
    print()

    # 4. Simulate an app restart: drop the agent, then build a brand-new one from the same session ID
    del agent
    agent = make_agent(model)
    print("--- New agent, same session ID: history loaded from disk ---")
    print_history(agent)
    print()

    print("=== Turn 3 (new agent, after reloading the session) ===")
    ask(agent, "Suggest an easy day trip out of the city for day three.")

    # 5. Clean up: delete_session() removes the session folder, so it can't be reloaded any more
    delete_saved_session()
    status = "is empty" if not any(SESSIONS_DIR.iterdir()) else "still has files"
    print(f"--- Session deleted (delete_session): {SESSIONS_DIR.name}/ {status} ---")

if __name__ == "__main__":
    # Model output contains characters like en dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    main()
