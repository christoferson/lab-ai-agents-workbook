import os
import sys
import time
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

    # 2. Build the model. Strands calls Bedrock directly through boto3 (no subprocess, no extra client
    #    to register), so the AWS profile and region go into a boto3 session.
    model = BedrockModel(
        model_id=model_id,
        boto_session=boto3.Session(profile_name=profile, region_name=region),
    )

    # 3. An agent is a model plus a system prompt. callback_handler=None turns off Strands' default
    #    handler, which would print the response as it streams in; here we print the result ourselves.
    #    The Agent code is the same whichever model it runs on.
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
    print(str(result).strip())  # str() joins the text blocks of the final message (gpt-oss's reasoning is left out)

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
