import os
import sys
import boto3
from strands import Agent
from strands.models.bedrock import BedrockModel

# Which model family to use, and the env var holding its Bedrock model ID (plus its default).
# All are called through Bedrock's Converse API, so the same BedrockModel class serves each one.
PROVIDERS = {
    "anthropic": ("STRANDS_MODEL_ID_ANTHROPIC", "global.anthropic.claude-sonnet-5"),
    "openai": ("STRANDS_MODEL_ID_OPENAI", "openai.gpt-oss-120b-1:0"),  # OpenAI's open-weight gpt-oss
    "amazon": ("STRANDS_MODEL_ID_AMAZON", "global.amazon.nova-2-lite-v1:0"),  # Amazon's own Nova
}


def ask(agent: Agent, prompt: str) -> str:
    """Send one message to an agent and return its reply. The agent appends both to agent.messages."""
    return str(agent(prompt)).strip()


def print_history(agent: Agent):
    """Print agent.messages as 'role: text'. Each message holds a list of content blocks, and only
    text blocks are shown (gpt-oss also stores its reasoning in a reasoningContent block)."""
    for i, message in enumerate(agent.messages, 1):
        text = "".join(block.get("text", "") for block in message["content"]).strip()
        print(f"  {i}. {message['role']}: {text}")


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

    # 2. One model, shared by both agents. The model holds no conversation, only the connection settings.
    model = BedrockModel(
        model_id=model_id,
        boto_session=boto3.Session(profile_name=profile, region_name=region),
    )
    system_prompt = (
        "You are a travel planner. Tailor suggestions to the traveler's destination and needs. "
        "If you are missing details, ask for them. Keep replies to one or two sentences."
    )

    def make_agent() -> Agent:
        return Agent(model=model, system_prompt=system_prompt, callback_handler=None)

    print("--- Agent ---")
    print(f"System prompt: {system_prompt}\n")

    # How memory works here: the Agent object is the conversation. Every call appends the prompt and
    # the reply to agent.messages, and the next call sends that list back to the model.
    # So "with history" means "ask the same agent again", and "without history" means "use another agent":
    # agent_with_conversation is asked both turns, agent_no_conversation only the follow-up question.
    agent_with_conversation = make_agent()
    agent_no_conversation = make_agent()

    first_prompt = (
        "I'm spending three relaxed days in Tokyo in April. "
        "I love nature, like gardens, parks and easy hikes, and I'd rather avoid crowds."
    )
    follow_up = "What should I do on my first morning?"

    # 3. Turn 1 on `agent_with_conversation`: share the trip details
    print("=== Turn 1: ask(agent_with_conversation, ...) ===")
    print(f"User:  {first_prompt}")
    print(f"Agent: {ask(agent_with_conversation, first_prompt)}")
    print(f"Messages held: {len(agent_with_conversation.messages)}\n")

    # 4. Turn 2 WITHOUT history: the same question to a second agent, whose messages list is empty
    print("=== Turn 2, WITHOUT history: ask(agent_no_conversation, ...) ===")
    print(f"User:  {follow_up}")
    print(f"Messages held before: {len(agent_no_conversation.messages)} (so it knows nothing)")
    print(f"Agent: {ask(agent_no_conversation, follow_up)}\n")

    # 5. Turn 2 WITH history: the same question to the agent from turn 1, which still holds that turn
    print("=== Turn 2, WITH history: ask(agent_with_conversation, ...) ===")
    print(f"User:  {follow_up}")
    print(f"Messages held before: {len(agent_with_conversation.messages)} (turn 1's prompt and reply)")
    print(f"Agent: {ask(agent_with_conversation, follow_up)}\n")

    # 6. agent.messages is a plain list in Converse format, so you can read (or edit) it directly
    print("--- History of agent_with_conversation (agent.messages) ---")
    print_history(agent_with_conversation)
    print("\n--- History of agent_no_conversation (agent.messages) ---")
    print_history(agent_no_conversation)

if __name__ == "__main__":
    # Model output contains characters like en dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    main()
