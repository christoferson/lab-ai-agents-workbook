import argparse
import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import boto3
from botocore.config import Config as BotocoreConfig
from openai import AsyncOpenAI
from openai.providers import bedrock
from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

# Which model family to use, and the env var holding its Bedrock model ID (plus its default).
# All are called through Bedrock's Converse API, so the same BedrockModel class serves each one.
PROVIDERS = {
    "anthropic": ("STRANDS_MODEL_ID_ANTHROPIC", "global.anthropic.claude-sonnet-5"),
    "openai": ("STRANDS_MODEL_ID_OPENAI", "openai.gpt-oss-120b-1:0"),  # OpenAI's open-weight gpt-oss
    "amazon": ("STRANDS_MODEL_ID_AMAZON", "global.amazon.nova-2-lite-v1:0"),  # Amazon's own Nova
}

# No year in the query: the agents get today's date, so they work out which autumn "this autumn" means
DEFAULT_QUERY = "Best places to see autumn leaves near Tokyo this autumn while avoiding the crowds"
HOW_MANY_SEARCHES = 3
MAX_REVIEW_ROUNDS = 3
OUTPUT_DIR = Path(__file__).with_name("reports") / "strands"

# Timeouts, so a stuck call fails instead of hanging the pipeline. Strands has no time limit for a
# whole agent run, so the script sets one around each run.
SEARCH_TIMEOUT = 90   # one web search request (usually 30-50s), retried once
MODEL_TIMEOUT = 120   # one Bedrock model call: seconds to wait for data, retried up to 2 more times
STEP_TIMEOUT = 600    # one agent's whole run, with all its model and tool calls

# Japan has no daylight saving time, so a fixed offset avoids needing the tzdata package on Windows
JST = timezone(timedelta(hours=9), "JST")


def today() -> str:
    """Today's date in Japan. Models don't know the current date, so every agent gets it in its system prompt."""
    return datetime.now(JST).strftime("%A, %d %B %Y")


# --- Structured outputs that hand data from one stage to the next. Each agent that returns one is
#     called with structured_output_model=..., as in the structured output example. ---

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


# --- Web search. The Converse API that BedrockModel uses has no web search, so web_search is a custom
#     tool. It calls Bedrock's own web search, the same one the OpenAI deep research example uses: a
#     Responses API request to a model on the bedrock-mantle endpoint, which searches and replies with
#     a summary and url_citation links. ---

TRACKING_PARAMS = {"msclkid", "gclid", "fbclid", "msockid", "click_product"}


def clean_url(url: str) -> str:
    """Drop ad-tracking query parameters that search results often carry, keeping ones the page needs."""
    parts = urlsplit(url)
    query_params = [(k, v) for k, v in parse_qsl(parts.query) if k not in TRACKING_PARAMS and not k.startswith("utm_")]
    return urlunsplit(parts._replace(query=urlencode(query_params)))


@dataclass
class BedrockWebSearch:
    """Runs Bedrock web search requests, and counts them: they are billed to the search model, so
    they don't appear in the agents' token usage."""
    client: AsyncOpenAI
    model_id: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    async def search(self, search_terms: str) -> str:
        response = await self.client.responses.create(
            model=self.model_id,
            instructions=f"Today is {today()} in Japan. Search the web and summarize what you find in under "
                         "150 words, with the facts a traveler needs. Note which year any dates refer to.",
            input=search_terms,
            # external_web_access=False keeps retrieval inside AWS, and works with AmazonBedrockFullAccess,
            # which does not grant bedrock-websearch:ExternalWebAccess
            tools=[{"type": "web_search", "search_context_size": "medium", "external_web_access": False}],
            # No tool_choice="required", which the other two versions use: in October 2026 tests every such
            # request timed out, while the default "auto" plus the instruction above still searched each time
        )
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

        sources = []
        messages = [item for item in response.output if item.type == "message"]
        for message in messages:
            for block in message.content:
                for ann in getattr(block, "annotations", None) or []:
                    # Some pages have no title, and Bedrock then puts the URL in the title too
                    url = clean_url(ann.url)
                    source = url if not ann.title or ann.title == ann.url else f"{ann.title}: {url}"
                    if ann.type == "url_citation" and source not in sources:
                        sources.append(source)
        # Clean the inline links too, since the agents copy URLs from the summary text
        summary = re.sub(r"https?://[^\s)\]]+", lambda m: clean_url(m.group()), response.output_text)
        return summary + "\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    def as_tool(self):
        """The search as a Strands tool. An async function works as a tool like a plain one."""
        @tool
        async def web_search(query: str) -> str:
            """Search the web and get a short summary of the results with source URLs.

            Args:
                query: The search terms to use.
            """
            return await self.search(query)
        return web_search


@tool
def save_report(title: str, markdown: str) -> str:
    """Save a finished research report as a Markdown file and return the file path.

    Args:
        title: Short report title, e.g. 'Quiet Autumn Leaves Near Tokyo'.
        markdown: The full report in Markdown.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


def tool_calls(agent: Agent) -> list[dict]:
    """The toolUse blocks in an agent's conversation, in order, leaving out the structured output tool."""
    blocks = [block for message in agent.messages for block in message["content"]]
    return [block["toolUse"] for block in blocks
            if "toolUse" in block and block["toolUse"]["name"] not in {SearchPlan.__name__, ReportData.__name__,
                                                                         FactCheck.__name__}]


# --- Agents: one role is a name, a system prompt and tools. A Strands Agent keeps its conversation and
#     can't run two calls at once, so the pipeline builds a fresh agent for every step. ---

@dataclass
class Role:
    name: str
    system_prompt: str
    tools: list = field(default_factory=list)


class StepTimeout(Exception):
    """An agent didn't finish its step within STEP_TIMEOUT seconds."""


@dataclass
class Strands:
    """Builds the agents on one shared model, and keeps every agent it built for the token summary."""
    model: BedrockModel
    agents: list[Agent] = field(default_factory=list)

    def agent(self, role: Role) -> Agent:
        """Factory: a fresh agent for one step of the pipeline."""
        agent = Agent(name=role.name, model=self.model, system_prompt=role.system_prompt, tools=role.tools,
                      callback_handler=None)
        self.agents.append(agent)
        return agent

    async def run(self, agent: Agent, prompt: str, **kwargs):
        """Run an agent, giving up after STEP_TIMEOUT seconds. wait_for cancels whatever the agent is
        waiting on, a model call or a tool call, so the step ends on time."""
        try:
            return await asyncio.wait_for(agent.invoke_async(prompt, **kwargs), STEP_TIMEOUT)
        except TimeoutError:
            raise StepTimeout(f"{agent.name} did not finish within {STEP_TIMEOUT}s") from None

    async def ask_for(self, schema: type[BaseModel], role: Role, prompt: str) -> tuple[BaseModel, Agent]:
        """Run a fresh agent for the role, and return the typed object it answered with, plus the agent."""
        agent = self.agent(role)
        result = await self.run(agent, prompt, structured_output_model=schema)
        return result.structured_output, agent

    def tokens(self) -> tuple[int, int]:
        usages = [agent.event_loop_metrics.accumulated_usage for agent in self.agents]
        return sum(u["inputTokens"] for u in usages), sum(u["outputTokens"] for u in usages)


def create_roles(web_search) -> dict[str, Role]:
    """Factory: return the roles of the research pipeline, keyed by role."""
    date_note = f"Today is {today()} in Japan."
    return {
        "planner": Role(
            "Planner",
            f"You are a travel research assistant. {date_note} Given a query, come up with "
            f"{HOW_MANY_SEARCHES} web searches that together best answer it. Give each search a "
            "different angle (for example forecast timing, lesser-known spots, or crowd and access tips) "
            "so they don't overlap, and keep each one within the area the query asks about.",
        ),
        "searcher": Role(
            "Searcher",
            f"{date_note} You research one search term. Always call web_search first, rather than answering "
            "from memory. Then write a concise summary of the results in under 120 words, with only facts "
            "useful to a traveler, noting which year any dates or forecasts refer to. End with the source "
            "URLs you relied on.",
            tools=[web_search],
        ),
        "writer": Role(
            "Writer",
            f"{date_note} You write a cohesive travel research report from a query and search summaries. "
            "Use only facts from the summaries. Aim for about 400 words in Markdown with short sections, "
            "and end with a Sources section listing the URLs you relied on.",
        ),
        "fact_checker": Role(
            "Fact Checker",
            f"{date_note} You fact-check a travel research report against the search summaries it was "
            "written from. Flag claims that contradict the summaries or each other, claims no summary "
            "supports, outdated information, and places described with the wrong location or in the wrong "
            "section. The summaries can be wrong too, so verify the most specific facts, such as distances, "
            "travel times and which place has which feature, by looking them up with web_search. "
            "Only report real problems; minor wording is fine.",
            tools=[web_search],
        ),
        "publisher": Role(
            "Publisher",
            "You save the report you are given with save_report using a clear title, "
            "then reply with where it was saved.",
            tools=[save_report],
        ),
    }


# --- The pipeline: plain Python code decides the order, agents do each step ---

async def plan_searches(strands: Strands, planner: Role, research_query: str) -> SearchPlan:
    print(f"1. {planner.name}: planning searches...")
    plan, _ = await strands.ask_for(SearchPlan, planner, f"Query: {research_query}")
    for i, item in enumerate(plan.searches, 1):
        print(f"   {i}. {item.query!r} - {item.reason}")
    print()
    return plan


async def search(strands: Strands, searcher: Role, item: SearchItem) -> str:
    """Research one search term with a fresh searcher agent and return its summary."""
    agent = strands.agent(searcher)
    try:
        result = await strands.run(agent, f"Search term: {item.query}\nReason for searching: {item.reason}")
    except StepTimeout as e:
        # One lost search shouldn't stop the report, so the writer gets the others
        print(f"   * {item.query!r}: {e}, skipped")
        return f"(No results for {item.query!r}: the search timed out.)"
    lookups = [call["input"].get("query") for call in tool_calls(agent)]
    print(f"   * {item.query!r}: {len(lookups)} web_search call(s): {lookups}")
    return str(result)


async def run_searches(strands: Strands, searcher: Role, plan: SearchPlan) -> list[str]:
    print(f"2. {searcher.name}: running {len(plan.searches)} searches in parallel...")
    summaries = await asyncio.gather(*(search(strands, searcher, item) for item in plan.searches))
    print()
    for item, summary in zip(plan.searches, summaries):
        print(f"   --- Summary for {item.query!r} ---")
        print("   " + summary.strip().replace("\n", "\n   ") + "\n")
    return summaries


async def write_report(strands: Strands, writer: Role, research_query: str, summaries: list[str]) -> ReportData:
    print(f"3. {writer.name}: writing the report...")
    report, _ = await strands.ask_for(ReportData, writer,
                                      f"Original query: {research_query}\nSummarized search results: {summaries}")
    print(f"   Summary: {report.short_summary}\n")
    return report


async def check_facts(strands: Strands, fact_checker: Role, report: ReportData, summaries: list[str],
                      round_no: int, fixed: list[FactIssue]) -> FactCheck:
    print(f"4. {fact_checker.name} (round {round_no}/{MAX_REVIEW_ROUNDS}): checking the report...")
    prompt = f"Report:\n{report.markdown_report}\n\nSearch summaries: {summaries}"
    if fixed:
        # Without this, a later round can re-flag a fixed claim or undo a correction
        done = "\n".join(f"- {i.claim} -> {i.correction}" for i in fixed)
        prompt += f"\n\nCorrections already applied in earlier rounds (don't re-flag them unless still wrong):\n{done}"
    check, agent = await strands.ask_for(FactCheck, fact_checker, prompt)
    # Show what was verified, so "0 issues" can be judged by what the checker actually looked up
    lookups = tool_calls(agent)
    print(f"   {len(check.issues)} issue(s) found, {len(lookups)} verification lookup(s)")
    for call in lookups:
        print(f"   - searched: {call['input'].get('query')}")
    for i, issue in enumerate(check.issues, 1):
        print(f"   {i}. Claim:      {issue.claim}")
        print(f"      Problem:    {issue.problem}")
        print(f"      Correction: {issue.correction}")
    print()
    return check


async def revise_report(strands: Strands, writer: Role, report: ReportData, check: FactCheck,
                        summaries: list[str]) -> ReportData:
    """Have a writer fix the fact checker's issues."""
    print(f"   {writer.name}: revising the report to fix {len(check.issues)} issue(s)...")
    issues = "\n".join(f"- {i.claim} -> {i.correction}" for i in check.issues)
    # Send the whole ReportData, so the writer keeps the follow-up questions and summarizes findings, not its edits
    revised, _ = await strands.ask_for(ReportData, writer,
                                       "Revise this report. Correct each flagged claim wherever it appears, "
                                       "without repeating information elsewhere, and keep everything else, "
                                       "including the follow-up questions. The short_summary must still summarize "
                                       "the findings, not describe your edits.\n\n"
                                       f"Issues:\n{issues}\n\nReport:\n{report.model_dump_json(indent=2)}\n\n"
                                       f"Search summaries: {summaries}")
    print(f"   Revised summary: {revised.short_summary}\n")
    return revised


async def review_report(strands: Strands, fact_checker: Role, writer: Role, report: ReportData,
                        summaries: list[str]) -> ReportData:
    """Check and revise until the fact checker finds no issues, for at most MAX_REVIEW_ROUNDS rounds."""
    fixed: list[FactIssue] = []
    for round_no in range(1, MAX_REVIEW_ROUNDS + 1):
        try:
            check = await check_facts(strands, fact_checker, report, summaries, round_no, fixed)
        except StepTimeout as e:
            # The report still exists, so publish it rather than lose the run
            print(f"   {e}. Publishing the report without this check.\n")
            return report
        if not check.issues:
            print(f"   Review passed in round {round_no}.\n")
            return report
        report = await revise_report(strands, writer, report, check, summaries)
        fixed += check.issues
    # The cap stops a checker that keeps finding something from looping (and costing) forever
    print(f"   Stopped after {MAX_REVIEW_ROUNDS} rounds; the last revision was not re-checked.\n")
    return report


async def publish_report(strands: Strands, publisher: Role, report: ReportData) -> str:
    print(f"5. {publisher.name}: saving the report...")
    agent = strands.agent(publisher)
    result = await strands.run(agent, report.markdown_report)
    blocks = [block for message in agent.messages for block in message["content"]]
    outputs = {b["toolResult"]["toolUseId"]: " ".join(p.get("text", "") for p in b["toolResult"]["content"])
               for b in blocks if "toolResult" in b}
    for call in tool_calls(agent):
        print(f"   Called {call['name']}(title={call['input'].get('title')!r}, "
              f"markdown=<{len(call['input'].get('markdown', ''))} chars>) -> {outputs.get(call['toolUseId'])}")
    print()
    return str(result)


async def main(research_query: str):

    # 1. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    provider = os.environ.get("STRANDS_MODEL_PROVIDER", "anthropic")
    if provider not in PROVIDERS:
        sys.exit(f"STRANDS_MODEL_PROVIDER must be one of {', '.join(PROVIDERS)}, not {provider!r}")
    model_env, default_model_id = PROVIDERS[provider]
    model_id = os.environ.get(model_env, default_model_id)
    # Converse rejects this model, but the Responses API web search runs on it
    search_model_id = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-5.6-luna")

    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print(f"Model Provider:     {provider} (STRANDS_MODEL_PROVIDER)")
    print(f"Model ID:           {model_id} ({model_env})")
    print(f"Search Model ID:    {search_model_id} (BEDROCK_MODEL_ID, runs Bedrock web search)")
    print("--------------------------\n")

    # 2. The web search tool, the pipeline's roles, and one shared model for every agent
    # Without a timeout the OpenAI client waits 10 minutes and retries twice. A search that still fails
    # reaches the agent as an error tool result, and the agent carries on without it.
    client = AsyncOpenAI(provider=bedrock(region=region), timeout=SEARCH_TIMEOUT, max_retries=1)
    web = BedrockWebSearch(client, search_model_id)
    roles = create_roles(web.as_tool())
    # Passing boto_client_config replaces Strands' default one, so it sets the read timeout itself
    model_config = BotocoreConfig(read_timeout=MODEL_TIMEOUT, connect_timeout=10,
                                  retries={"total_max_attempts": 3, "mode": "standard"})
    strands = Strands(BedrockModel(model_id=model_id, boto_client_config=model_config,
                                   boto_session=boto3.Session(profile_name=profile, region_name=region)))

    print("--- Pipeline ---")
    print(f"1. {roles['planner'].name}:      query -> {SearchPlan.__name__} (structured_output_model)")
    print(f"2. {roles['searcher'].name}:     each search -> summary, using web_search, in parallel")
    print(f"3. {roles['writer'].name}:       summaries -> {ReportData.__name__} (structured_output_model)")
    print(f"4. {roles['fact_checker'].name}: report -> {FactCheck.__name__} (structured_output_model), using web_search;")
    print(f"                   while it finds issues, a Writer revises and it re-checks (up to {MAX_REVIEW_ROUNDS} rounds)")
    print(f"5. {roles['publisher'].name}:    report -> Markdown file, using the save_report tool")
    print("Every step runs a fresh Agent, so no step sees another step's conversation.")
    print(f"Date given to agents: {today()} (Japan time)")
    print(f"web_search: Bedrock web search on {search_model_id}, external_web_access=False")
    print(f"Timeouts: web search {SEARCH_TIMEOUT}s (1 retry), model call {MODEL_TIMEOUT}s (2 retries), "
          f"agent step {STEP_TIMEOUT}s\n")

    print("--- Query ---")
    print(f"{research_query}\n")

    print(f"Running on {model_id} via Amazon Bedrock...\n")
    start = time.perf_counter()
    plan = await plan_searches(strands, roles["planner"], research_query)
    summaries = await run_searches(strands, roles["searcher"], plan)
    if web.calls == 0:
        # Searchers still reply when every search fails, and a writer would then write from memory
        sys.exit("Stopped: no web search succeeded, so there is nothing to base a report on.")
    report = await write_report(strands, roles["writer"], research_query, summaries)
    report = await review_report(strands, roles["fact_checker"], roles["writer"], report, summaries)
    await publish_report(strands, roles["publisher"], report)
    duration = time.perf_counter() - start

    print("--- Report ---")
    print(report.markdown_report)
    print("\n--- Follow-up questions ---")
    for question in report.follow_up_questions:
        print(f"* {question}")

    # 3. Two bills: the agents' tokens on the Strands model, and web search billed to the search model
    tokens_in, tokens_out = strands.tokens()
    print("\n--- Run Summary ---")
    print(f"Duration:     {duration:.1f}s")
    print(f"Agent runs:   {len(strands.agents)}")
    print(f"Agent tokens: {tokens_in} in, {tokens_out} out on {model_id}")
    print(f"Web searches: {web.calls} on {search_model_id}, {web.input_tokens} in / {web.output_tokens} out tokens "
          "(not in the agent tokens)")

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
