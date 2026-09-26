import argparse
import asyncio
import os
import re
from pathlib import Path
from openai import AsyncOpenAI
from openai.providers import bedrock
from pydantic import BaseModel, Field
# Import the explicit tracing control from the agents SDK
from agents import (
    Agent, ModelSettings, Runner, WebSearchTool, function_tool, set_default_openai_client, set_tracing_disabled,
)

DEFAULT_QUERY = "Best places to see autumn leaves near Tokyo in 2026 while avoiding the crowds"
HOW_MANY_SEARCHES = 3
OUTPUT_DIR = Path(__file__).with_name("reports")


# --- Tools ---
# Bedrock hosts the web search tool on the bedrock-mantle endpoint (the default for the bedrock() provider).
# external_web_access=False keeps retrieval inside AWS: Search uses the Bedrock web index and Fetch its cache.
# It also works with AmazonBedrockFullAccess, which does not grant bedrock-websearch:ExternalWebAccess.
web_search = WebSearchTool(search_context_size="low", external_web_access=False)

# The SDK always sends "filters" and "user_location" (as null), which Bedrock rejects with a 400.
# extra_body overrides top-level request keys, so it replaces the SDK's tools list with just the supported fields.
BEDROCK_WEB_SEARCH = {
    "type": "web_search",
    "search_context_size": web_search.search_context_size,
    "external_web_access": web_search.external_web_access,
}


@function_tool
def save_report(title: str, markdown: str) -> str:
    """Save a finished research report as a Markdown file and return the file path.

    Args:
        title: Short report title, e.g. 'Quiet Autumn Leaves Near Tokyo'.
        markdown: The full report in Markdown.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "report"
    path = OUTPUT_DIR / f"{slug}.md"
    # The writer usually starts the report with its own H1, so only add one when it's missing
    body = markdown.strip() if markdown.lstrip().startswith("# ") else f"# {title}\n\n{markdown.strip()}"
    path.write_text(body + "\n", encoding="utf-8")
    return f"Saved to {path.relative_to(Path(__file__).parent.parent)}"


# --- Structured outputs that hand data from one stage to the next ---

class SearchItem(BaseModel):
    reason: str = Field(description="Why this search helps answer the query")
    query: str = Field(description="The search terms to use")


class SearchPlan(BaseModel):
    searches: list[SearchItem] = Field(description="The searches to perform to best answer the query")


class ReportData(BaseModel):
    short_summary: str = Field(description="A 2-3 sentence summary of the findings")
    markdown_report: str = Field(description="The final report in Markdown, ending with a Sources section")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")


def create_agents(model_id: str) -> dict[str, Agent]:
    """Factory: return the four agents of the research pipeline, keyed by role."""
    return {
        "planner": Agent(
            name="Planner",
            instructions=f"You are a travel research assistant. Given a query, come up with {HOW_MANY_SEARCHES} "
                         "web searches that together best answer it.",
            model=model_id,
            output_type=SearchPlan,
        ),
        "searcher": Agent(
            name="Searcher",
            instructions="You search the web for one search term and write a concise summary of the results "
                         "in under 120 words. Capture only facts useful to a traveler.",
            model=model_id,
            tools=[web_search],
            # Force a search, so the summary is grounded in current results rather than the model's memory
            model_settings=ModelSettings(tool_choice="required", extra_body={"tools": [BEDROCK_WEB_SEARCH]}),
        ),
        "writer": Agent(
            name="Writer",
            instructions="You write a cohesive travel research report from a query and search summaries. "
                         "Use only facts from the summaries. Aim for about 400 words in Markdown with short sections, "
                         "and end with a Sources section listing the URLs you relied on.",
            model=model_id,
            output_type=ReportData,
        ),
        "publisher": Agent(
            name="Publisher",
            instructions="You save the report you are given with save_report using a clear title, "
                         "then reply with where it was saved.",
            model=model_id,
            tools=[save_report],
        ),
    }


def extract_sources(result) -> list[str]:
    """Collect the url_citation annotations Bedrock attaches to the searcher's answer."""
    sources = []
    for item in result.new_items:
        if item.type == "message_output_item":
            for block in item.raw_item.content:
                for ann in getattr(block, "annotations", None) or []:
                    if ann.type == "url_citation" and f"{ann.title}: {ann.url}" not in sources:
                        sources.append(f"{ann.title}: {ann.url}")
    return sources


# --- The pipeline: plain Python code decides the order, agents do each step ---

async def plan_searches(planner: Agent, query: str) -> SearchPlan:
    print(f"1. {planner.name}: planning searches...")
    result = await Runner.run(planner, f"Query: {query}")
    plan = result.final_output
    for i, item in enumerate(plan.searches, 1):
        print(f"   {i}. {item.query!r} - {item.reason}")
    print()
    return plan


async def search(searcher: Agent, item: SearchItem) -> str:
    """Run one web search and return its summary followed by the sources it cited."""
    result = await Runner.run(searcher, f"Search term: {item.query}\nReason for searching: {item.reason}")
    searches = sum(1 for i in result.new_items if i.type == "tool_call_item")
    sources = extract_sources(result)
    print(f"   * {item.query!r}: {searches} web search call(s), {len(sources)} source(s) cited")
    return result.final_output + "\nSources:\n" + "\n".join(f"- {s}" for s in sources)


async def run_searches(searcher: Agent, plan: SearchPlan) -> list[str]:
    print(f"2. {searcher.name}: running {len(plan.searches)} searches in parallel...")
    summaries = await asyncio.gather(*(search(searcher, item) for item in plan.searches))
    print()
    for item, summary in zip(plan.searches, summaries):
        print(f"   --- Summary for {item.query!r} ---")
        print("   " + summary.replace("\n", "\n   ") + "\n")
    return summaries


async def write_report(writer: Agent, query: str, summaries: list[str]) -> ReportData:
    print(f"3. {writer.name}: writing the report...")
    result = await Runner.run(writer, f"Original query: {query}\nSummarized search results: {summaries}")
    report = result.final_output
    print(f"   Summary: {report.short_summary}\n")
    return report


async def publish_report(publisher: Agent, report: ReportData) -> str:
    print(f"4. {publisher.name}: saving the report...")
    result = await Runner.run(publisher, report.markdown_report)
    print(f"   {result.final_output}\n")
    return result.final_output


async def main(query: str):

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

    # 5. Build the pipeline's agents
    agents = create_agents(model_id)

    print("--- Pipeline ---")
    print(f"1. {agents['planner'].name}:   query -> {SearchPlan.__name__} (structured output)")
    print(f"2. {agents['searcher'].name}:  each search -> summary with sources, using Bedrock web search, in parallel")
    print(f"3. {agents['writer'].name}:    summaries -> {ReportData.__name__} (structured output)")
    print(f"4. {agents['publisher'].name}: report -> Markdown file, using the save_report tool")
    print(f"Web search: search_context_size={web_search.search_context_size!r}, "
          f"external_web_access={web_search.external_web_access}\n")

    print("--- Query ---")
    print(f"{query}\n")

    print(f"Running on {model_id} via Amazon Bedrock...\n")
    plan = await plan_searches(agents["planner"], query)
    summaries = await run_searches(agents["searcher"], plan)
    report = await write_report(agents["writer"], query, summaries)
    await publish_report(agents["publisher"], report)

    print("--- Report ---")
    print(report.markdown_report)
    print("\n--- Follow-up questions ---")
    for question in report.follow_up_questions:
        print(f"* {question}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research a question on the web with a multi-agent pipeline.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="the research question")
    args = parser.parse_args()
    asyncio.run(main(args.query))
