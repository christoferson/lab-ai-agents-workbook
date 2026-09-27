import asyncio
import os
import sys
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, StreamEvent, query


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

    # 2. Configure the agent. include_partial_messages=True makes query() also yield StreamEvent
    #    messages: the raw stream events from the model, including each piece of text as it is generated.
    system_prompt = "You explain AI agent concepts clearly and concisely to developers."
    options = ClaudeAgentOptions(
        model=model_id,
        system_prompt=system_prompt,
        tools=[],  # no built-in tools (file access, shell, ...): a plain question-and-answer agent
        max_turns=1,
        setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
        include_partial_messages=True,
        env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region},
    )

    print("--- Agent ---")
    print(f"System prompt: {system_prompt}\n")

    prompt = "List the 5 core building blocks of an AI agent, one line each."
    print("--- Prompt ---")
    print(f"{prompt}\n")

    print(f"Streaming response from {model_id} via Amazon Bedrock...\n")

    # 3. Print each text delta as it arrives. The complete AssistantMessage still comes afterwards,
    #    so it is skipped here to avoid printing the answer twice.
    print("--- Agent Response (streaming) ---")
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, StreamEvent):
            event = message.event
            if event.get("type") == "content_block_delta" and event["delta"].get("type") == "text_delta":
                print(event["delta"]["text"], end="", flush=True)
        elif isinstance(message, ResultMessage):
            print("\n\n--- Run Summary ---")
            print(f"Status:   {'error' if message.is_error else 'success'}")
            print(f"Duration: {message.duration_ms / 1000:.1f}s")
            if message.total_cost_usd is not None:
                print(f"Cost:     ${message.total_cost_usd:.4f}")

if __name__ == "__main__":
    # Model output contains characters like en dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
