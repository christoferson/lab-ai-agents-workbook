import asyncio
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


async def main():

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

    # 2. The same model and agent as the basic example. BedrockModel streams by default (ConverseStream);
    #    callback_handler=None stops Strands printing the stream itself, so the loop below can.
    model = BedrockModel(
        model_id=model_id,
        boto_session=boto3.Session(profile_name=profile, region_name=region),
    )
    system_prompt = "You explain AI agent concepts clearly and concisely to developers."
    agent = Agent(model=model, system_prompt=system_prompt, callback_handler=None)

    print("--- Agent ---")
    print(f"System prompt: {system_prompt}\n")

    prompt = "List the 5 core building blocks of an AI agent, one line each."
    print("--- Prompt ---")
    print(f"{prompt}\n")

    print(f"Streaming response from {model_id} via Amazon Bedrock...\n")

    # 3. stream_async() yields plain dicts. A text chunk has a "data" key, and the last event has a
    #    "result" key holding the same AgentResult that agent(prompt) returns.
    print("--- Agent Response (streaming) ---")
    start = time.perf_counter()
    first_chunk_at = None
    result = None
    async for event in agent.stream_async(prompt):
        if "data" in event:
            first_chunk_at = first_chunk_at or time.perf_counter()
            print(event["data"], end="", flush=True)
        elif "result" in event:
            result = event["result"]
    duration = time.perf_counter() - start

    # 4. Strands reports token usage rather than a cost
    usage = result.metrics.accumulated_usage
    print("\n\n--- Run Summary ---")
    print(f"Stop reason: {result.stop_reason}")
    print(f"First text:  {first_chunk_at - start:.1f}s")  # how long streaming saves before anything shows
    print(f"Duration:    {duration:.1f}s")
    print(f"Tokens:      {usage['inputTokens']} in, {usage['outputTokens']} out")

if __name__ == "__main__":
    # Model output contains characters like en dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
