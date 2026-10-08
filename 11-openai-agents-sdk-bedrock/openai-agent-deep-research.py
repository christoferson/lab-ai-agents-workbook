import argparse
import asyncio
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from openai import APITimeoutError, AsyncOpenAI
from openai.providers import bedrock
from pydantic import BaseModel, Field
# Import the explicit tracing control from the agents SDK
from agents import (
    Agent, ModelSettings, Runner, WebSearchTool, function_tool, set_default_openai_client, set_tracing_disabled,
)

# No year in the query: the agents get today's date, so they work out which autumn "this autumn" means
DEFAULT_QUERY = "Best places to see autumn leaves near Tokyo this autumn while avoiding the crowds"
HOW_MANY_SEARCHES = 3
MAX_REVIEW_ROUNDS = 3
OUTPUT_DIR = Path(__file__).with_name("reports")

# Timeouts, so a stuck call fails instead of hanging the pipeline. The SDK has no time limit for a
# whole run, so the script sets one around each run.
MODEL_TIMEOUT = 180  # one model request, including any web searches it makes (usually 30-60s), retried once
STEP_TIMEOUT = 600   # one agent's whole run, with all its model and tool calls

# Japan has no daylight saving time, so a fixed offset avoids needing the tzdata package on Windows
JST = timezone(timedelta(hours=9), "JST")


def today() -> str:
    """Today's date in Japan. Models don't know the current date, so every agent gets it in its instructions."""
    return datetime.now(JST).strftime("%A, %d %B %Y")


# --- Tools ---
# Bedrock hosts the web search tool on the bedrock-mantle endpoint (the default for the bedrock() provider).
# external_web_access=False keeps retrieval inside AWS: Search uses the Bedrock web index and Fetch its cache.
# It also works with AmazonBedrockFullAccess, which does not grant bedrock-websearch:ExternalWebAccess.
web_search = WebSearchTool(search_context_size="medium", external_web_access=False)

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
    # Research on the same topic often gets the same title, so number new files instead of overwriting old reports
    n = 2
    while path.exists():
        path = OUTPUT_DIR / f"{slug}-{n}.md"
        n += 1
    # The writer usually starts the report with its own H1, so only add one when it's missing
    body = markdown.strip() if markdown.lstrip().startswith("# ") else f"# {title}\n\n{markdown.strip()}"
    path.write_text(body + "\n", encoding="utf-8")
    return f"Saved to {path.relative_to(Path(__file__).parent.parent).as_posix()}"


# --- Structured outputs that hand data from one stage to the next ---

class SearchItem(BaseModel):
    reason: str = Field(description="Why this search helps answer the query")
    query: str = Field(description="The search terms to use")


class SearchPlan(BaseModel):
    searches: list[SearchItem] = Field(description="The searches to perform to best answer the query")


class ReportData(BaseModel):
    short_summary: str = Field(description="A 2-3 sentence summary of the findings")
    markdown_report: str = Field(description="The final report in Markdown, ending with a Sources section")
    follow_up_questions: list[str] = Field(
        description="Suggested topics to research further, as research questions, not questions to the traveler")


class FactIssue(BaseModel):
    claim: str = Field(description="The claim in the report, quoted or closely paraphrased")
    problem: str = Field(description="Why it is wrong, unsupported, contradictory or outdated")
    correction: str = Field(description="What the report should say instead, with a source URL if you have one")


class FactCheck(BaseModel):
    issues: list[FactIssue] = Field(description="Problems found in the report; empty if it is accurate")


def create_agents(model_id: str) -> dict[str, Agent]:
    """Factory: return the agents of the research pipeline, keyed by role."""
    date_note = f"Today is {today()} in Japan."
    return {
        "planner": Agent(
            name="Planner",
            instructions=f"You are a travel research assistant. {date_note} Given a query, come up with "
                         f"{HOW_MANY_SEARCHES} web searches that together best answer it. Give each search a "
                         "different angle (for example forecast timing, lesser-known spots, or crowd and access tips) "
                         "so they don't overlap, and keep each one within the area the query asks about.",
            model=model_id,
            output_type=SearchPlan,
        ),
        "searcher": Agent(
            name="Searcher",
            instructions=f"{date_note} You search the web for one search term. Always search first, rather than "
                         "answering from memory. Then write a concise summary of the results in under 120 words. Capture only facts useful to a traveler, and note which "
                         "year any dates or forecasts refer to.",
            model=model_id,
            tools=[web_search],
            # No tool_choice="required" to force the search: in October 2026 tests every such request timed
            # out, while the default "auto" plus the instruction to search first still searched each time
            model_settings=ModelSettings(extra_body={"tools": [BEDROCK_WEB_SEARCH]}),
        ),
        "writer": Agent(
            name="Writer",
            instructions=f"{date_note} You write a cohesive travel research report from a query and search "
                         "summaries. Use only facts from the summaries. Aim for about 400 words in Markdown with "
                         "short sections, and end with a Sources section listing the URLs you relied on.",
            model=model_id,
            output_type=ReportData,
        ),
        "fact_checker": Agent(
            name="Fact Checker",
            instructions=f"{date_note} You fact-check a travel research report against the search summaries it "
                         "was written from. Flag claims that contradict the summaries or each other, claims no "
                         "summary supports, outdated information, and places described with the wrong location or "
                         "in the wrong section. The summaries can be wrong too, so always use web search to verify "
                         "the most specific facts, such as distances, travel times and which place has which "
                         "feature. Only report real problems; minor wording is fine.",
            model=model_id,
            tools=[web_search],
            output_type=FactCheck,
            model_settings=ModelSettings(extra_body={"tools": [BEDROCK_WEB_SEARCH]}),
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
                    url = clean_url(ann.url)
                    # Some pages have no title, and Bedrock then puts the URL in the title too
                    source = url if not ann.title or ann.title == ann.url else f"{ann.title}: {url}"
                    if ann.type == "url_citation" and source not in sources:
                        sources.append(source)
    return sources


TRACKING_PARAMS = {"msclkid", "gclid", "fbclid", "msockid", "click_product"}


def clean_url(url: str) -> str:
    """Drop ad-tracking query parameters that search results often carry, keeping ones the page needs."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k not in TRACKING_PARAMS and not k.startswith("utm_")]
    return urlunsplit(parts._replace(query=urlencode(query)))


# --- The pipeline: plain Python code decides the order, agents do each step ---

class StepTimeout(Exception):
    """An agent didn't finish its step in time."""


async def run_step(agent: Agent, prompt: str):
    """Runner.run, giving up after STEP_TIMEOUT seconds, or when a model request times out after its retry.
    wait_for cancels the request the agent is waiting on, so the step ends on time."""
    try:
        return await asyncio.wait_for(Runner.run(agent, prompt), STEP_TIMEOUT)
    except TimeoutError:
        raise StepTimeout(f"{agent.name} did not finish within {STEP_TIMEOUT}s") from None
    except APITimeoutError:
        raise StepTimeout(f"{agent.name}'s model request timed out after {MODEL_TIMEOUT}s") from None


async def plan_searches(planner: Agent, query: str) -> SearchPlan:
    print(f"1. {planner.name}: planning searches...")
    result = await run_step(planner, f"Query: {query}")
    plan = result.final_output
    for i, item in enumerate(plan.searches, 1):
        print(f"   {i}. {item.query!r} - {item.reason}")
    print()
    return plan


async def search(searcher: Agent, item: SearchItem) -> tuple[str, int]:
    """Run one web search and return its summary followed by the sources it cited, and how many
    searches it made."""
    try:
        result = await run_step(searcher, f"Search term: {item.query}\nReason for searching: {item.reason}")
    except StepTimeout as e:
        # One lost search shouldn't stop the report, so the writer gets the others
        print(f"   * {item.query!r}: {e}, skipped")
        return f"(No results for {item.query!r}: the search timed out.)", 0
    searches = sum(1 for i in result.new_items if i.type == "tool_call_item")
    sources = extract_sources(result)
    print(f"   * {item.query!r}: {searches} web search call(s), {len(sources)} source(s) cited")
    # Clean the inline links too, since the writer copies URLs from the summary text
    summary = re.sub(r"https?://[^\s)\]]+", lambda m: clean_url(m.group()), result.final_output)
    return summary + "\nSources:\n" + "\n".join(f"- {s}" for s in sources), searches


async def run_searches(searcher: Agent, plan: SearchPlan) -> list[str]:
    print(f"2. {searcher.name}: running {len(plan.searches)} searches in parallel...")
    results = await asyncio.gather(*(search(searcher, item) for item in plan.searches))
    if sum(searches for _, searches in results) == 0:
        # Without a single search the writer would write from memory
        sys.exit("Stopped: no web search succeeded, so there is nothing to base a report on.")
    summaries = [summary for summary, _ in results]
    print()
    for item, summary in zip(plan.searches, summaries):
        print(f"   --- Summary for {item.query!r} ---")
        print("   " + summary.replace("\n", "\n   ") + "\n")
    return summaries


async def write_report(writer: Agent, query: str, summaries: list[str]) -> ReportData:
    print(f"3. {writer.name}: writing the report...")
    result = await run_step(writer, f"Original query: {query}\nSummarized search results: {summaries}")
    report = result.final_output
    print(f"   Summary: {report.short_summary}\n")
    return report


async def check_facts(fact_checker: Agent, report: ReportData, summaries: list[str], round_no: int,
                      fixed: list[FactIssue]) -> FactCheck:
    print(f"4. {fact_checker.name} (round {round_no}/{MAX_REVIEW_ROUNDS}): checking the report...")
    prompt = f"Report:\n{report.markdown_report}\n\nSearch summaries: {summaries}"
    if fixed:
        # Without this, a later round can re-flag a fixed claim or undo a correction
        done = "\n".join(f"- {i.claim} -> {i.correction}" for i in fixed)
        prompt += f"\n\nCorrections already applied in earlier rounds (don't re-flag them unless still wrong):\n{done}"
    result = await run_step(fact_checker, prompt)
    check = result.final_output
    searches = [i.raw_item for i in result.new_items if i.type == "tool_call_item"]
    print(f"   {len(check.issues)} issue(s) found, {len(searches)} verification search(es)")
    # Show what was verified, so "0 issues" can be judged by what the checker actually looked up
    for call in searches:
        action = getattr(call, "action", None)
        print(f"   - searched: {getattr(action, 'query', None) or getattr(action, 'url', None) or '(no details)'}")
    for i, issue in enumerate(check.issues, 1):
        print(f"   {i}. Claim:      {issue.claim}")
        print(f"      Problem:    {issue.problem}")
        print(f"      Correction: {issue.correction}")
    print()
    return check


async def revise_report(writer: Agent, report: ReportData, check: FactCheck, summaries: list[str]) -> ReportData:
    """Have the writer fix the fact checker's issues."""
    print(f"   {writer.name}: revising the report to fix {len(check.issues)} issue(s)...")
    issues = "\n".join(f"- {i.claim} -> {i.correction}" for i in check.issues)
    # Send the whole ReportData, so the writer keeps the follow-up questions and summarizes findings, not its edits
    result = await run_step(writer, "Revise this report. Correct each flagged claim wherever it appears, "
                                      "without repeating information elsewhere, and keep everything else, "
                                      "including the follow-up questions. The short_summary must still summarize "
                                      "the findings, not describe your edits.\n\n"
                                      f"Issues:\n{issues}\n\nReport:\n{report.model_dump_json(indent=2)}\n\n"
                                      f"Search summaries: {summaries}")
    revised = result.final_output
    print(f"   Revised summary: {revised.short_summary}\n")
    return revised


async def review_report(fact_checker: Agent, writer: Agent, report: ReportData, summaries: list[str]) -> ReportData:
    """Check and revise until the fact checker finds no issues, for at most MAX_REVIEW_ROUNDS rounds."""
    fixed: list[FactIssue] = []
    for round_no in range(1, MAX_REVIEW_ROUNDS + 1):
        try:
            check = await check_facts(fact_checker, report, summaries, round_no, fixed)
        except StepTimeout as e:
            # The report still exists, so publish it rather than lose the run
            print(f"   {e}. Publishing the report without this check.\n")
            return report
        if not check.issues:
            print(f"   Review passed in round {round_no}.\n")
            return report
        report = await revise_report(writer, report, check, summaries)
        fixed += check.issues
    # The cap stops a checker that keeps finding something from looping (and costing) forever
    print(f"   Stopped after {MAX_REVIEW_ROUNDS} rounds; the last revision was not re-checked.\n")
    return report


async def publish_report(publisher: Agent, report: ReportData) -> str:
    print(f"5. {publisher.name}: saving the report...")
    result = await run_step(publisher, report.markdown_report)
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
    #    Without a timeout the client waits 10 minutes for a request and retries twice.
    bedrock_client = AsyncOpenAI(
        provider=bedrock(
            region=region
        ),
        timeout=MODEL_TIMEOUT,
        max_retries=1,
    )

    # 4. Register your Bedrock client as default for the OpenAI Agents SDK
    set_default_openai_client(bedrock_client)

    # 5. Build the pipeline's agents
    agents = create_agents(model_id)

    print("--- Pipeline ---")
    print(f"1. {agents['planner'].name}:      query -> {SearchPlan.__name__} (structured output)")
    print(f"2. {agents['searcher'].name}:     each search -> summary with sources, using web search, in parallel")
    print(f"3. {agents['writer'].name}:       summaries -> {ReportData.__name__} (structured output)")
    print(f"4. {agents['fact_checker'].name}: report -> {FactCheck.__name__} (structured output, may use web search);")
    print(f"                   while it finds issues, the Writer revises and it re-checks "
          f"(up to {MAX_REVIEW_ROUNDS} rounds)")
    print(f"5. {agents['publisher'].name}:    report -> Markdown file, using the save_report tool")
    print(f"Date given to agents: {today()} (Japan time)")
    print(f"Web search: search_context_size={web_search.search_context_size!r}, "
          f"external_web_access={web_search.external_web_access}")
    print(f"Timeouts: model request {MODEL_TIMEOUT}s (1 retry), agent step {STEP_TIMEOUT}s\n")

    print("--- Query ---")
    print(f"{query}\n")

    print(f"Running on {model_id} via Amazon Bedrock...\n")
    plan = await plan_searches(agents["planner"], query)
    summaries = await run_searches(agents["searcher"], plan)
    report = await write_report(agents["writer"], query, summaries)
    report = await review_report(agents["fact_checker"], agents["writer"], report, summaries)
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
    # Web content contains characters like em dashes that Windows code pages (e.g. cp932) can't encode
    # when output is redirected to a file, so always write UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main(args.query))
    except StepTimeout as e:
        # The planner, writer and publisher have no fallback, so the run stops with a clear reason
        sys.exit(f"\nStopped: {e}.")
