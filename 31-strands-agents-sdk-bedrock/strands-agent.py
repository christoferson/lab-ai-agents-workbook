import os
import sys
import time
import boto3
from strands import Agent
from strands.models.bedrock import BedrockModel


def main():

    # 1. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    model_id = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", "global.anthropic.claude-sonnet-5")

    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print(f"Claude Model ID:    {model_id}")
    print("--------------------------\n")

    # 2. Build the model. Strands calls Bedrock directly through boto3 (no subprocess, no extra client
    #    to register), so the AWS profile and region go into a boto3 session.
    model = BedrockModel(
        model_id=model_id,
        boto_session=boto3.Session(profile_name=profile, region_name=region),
    )

    # 3. An agent is a model plus a system prompt. callback_handler=None turns off Strands' default
    #    handler, which would print the response as it streams in; here we print the result ourselves.
    system_prompt = "You explain AI agent concepts clearly and concisely to developers."
    agent = Agent(model=model, system_prompt=system_prompt, callback_handler=None)

    print("--- Agent ---")
    print(f"System prompt: {system_prompt}\n")

    # 4. Send the prompt. Calling the agent runs its loop synchronously and returns an AgentResult.
    prompt = "In 3 sentences, what is the Strands Agents SDK and how does it differ from calling the Bedrock API directly?"
    print("--- Prompt ---")
    print(f"{prompt}\n")

    print(f"Running agent on {model_id} via Amazon Bedrock...\n")
    start = time.perf_counter()
    result = agent(prompt)
    duration = time.perf_counter() - start

    print("--- Agent Response ---")
    print(str(result).strip())  # str() joins the text blocks of the final message

    # 5. Strands reports token usage rather than a cost
    usage = result.metrics.accumulated_usage
    print("\n--- Run Summary ---")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Cycles:      {result.metrics.cycle_count}")  # one model call (plus tool calls) per cycle
    print(f"Duration:    {duration:.1f}s")
    print(f"Tokens:      {usage['inputTokens']} in, {usage['outputTokens']} out")

if __name__ == "__main__":
    # Model output contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    main()
