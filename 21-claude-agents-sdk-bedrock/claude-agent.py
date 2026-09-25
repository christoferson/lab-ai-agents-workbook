import asyncio
import os
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query


async def main():

    # 1. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    model_id = os.environ.get("CLAUDE_MODEL_ID", "global.anthropic.claude-sonnet-5")

    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print(f"Claude Model ID:    {model_id}")
    print("--------------------------\n")

    # 2. Configure the agent. The SDK runs the bundled Claude Code CLI as a subprocess;
    #    CLAUDE_CODE_USE_BEDROCK=1 tells it to call Amazon Bedrock with your AWS credentials.
    system_prompt = "You explain AI agent concepts clearly and concisely to developers."
    options = ClaudeAgentOptions(
        model=model_id,
        system_prompt=system_prompt,
        tools=[],  # no built-in tools (file access, shell, ...): a plain question-and-answer agent
        max_turns=1,
        env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": profile, "AWS_REGION": region},
    )

    print("--- Agent ---")
    print(f"System prompt: {system_prompt}\n")

    # 3. Send the prompt; query() streams back messages as the agent works
    prompt = "In 3 sentences, what is the Claude Agent SDK and how does it differ from calling the Claude API directly?"
    print("--- Prompt ---")
    print(f"{prompt}\n")

    print(f"Running agent on {model_id} via Amazon Bedrock...\n")
    print("--- Agent Response ---")
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)
        elif isinstance(message, ResultMessage):
            # The final message summarizes the run
            print("\n--- Run Summary ---")
            print(f"Status:   {'error' if message.is_error else 'success'}")
            print(f"Turns:    {message.num_turns}")
            print(f"Duration: {message.duration_ms / 1000:.1f}s")
            if message.total_cost_usd is not None:
                print(f"Cost:     ${message.total_cost_usd:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
