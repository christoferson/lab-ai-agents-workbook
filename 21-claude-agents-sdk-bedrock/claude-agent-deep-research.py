import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ResultError, ResultMessage, SystemMessage, ToolResultBlock, ToolUseBlock,
    UserMessage, create_sdk_mcp_server, delete_session, query, tool,
)
from openai import APIError, AsyncOpenAI
from openai.providers import bedrock
from pydantic import BaseModel, Field

# No year in the query: the agents get today's date, so they work out which autumn "this autumn" means
DEFAULT_QUERY = "Best places to see autumn leaves near Tokyo this autumn while avoiding the crowds"
HOW_MANY_SEARCHES = 3
MAX_REVIEW_ROUNDS = 3
OUTPUT_DIR = Path(__file__).with_name("reports") / "claude"
SERVER_NAME = "research"
WEB_FETCH = "WebFetch"  # Claude Code's built-in tool for reading one page; it works on Bedrock

# Timeouts, so a stuck call fails instead of hanging the pipeline. The SDK has no time limit for a
# whole run, so the script sets one around each run.
SEARCH_TIMEOUT = 90   # one web search request (usually 30-50s), retried once
MODEL_TIMEOUT = 120   # one Claude model request, retried up to MODEL_RETRIES times
MODEL_RETRIES = 2
STEP_TIMEOUT = 600    # one agent's whole run, with all its model and tool calls

# Japan has no daylight saving time, so a fixed offset avoids needing the tzdata package on Windows
JST = timezone(timedelta(hours=9), "JST")


def today() -> str:
    """Today's date in Japan. Models don't know the current date, so every agent gets it in its system prompt."""
    return datetime.now(JST).strftime("%A, %d %B %Y")


# --- Structured outputs that hand data from one stage to the next. The Claude Agent SDK has no
#     output_type, so, as in the structured output example, each agent answers by calling a tool
#     whose input schema is the model. ---

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


@tool("submit_search_plan", "Submit your search plan.", SearchPlan.model_json_schema())
async def submit_search_plan(args):
    return {"content": [{"type": "text", "text": "Search plan recorded."}]}


@tool("submit_report", "Submit the finished report.", ReportData.model_json_schema())
async def submit_report(args):
    return {"content": [{"type": "text", "text": "Report recorded."}]}


@tool("submit_fact_check", "Submit your fact check, with an empty issues list if the report is accurate.",
      FactCheck.model_json_schema())
async def submit_fact_check(args):
    return {"content": [{"type": "text", "text": "Fact check recorded."}]}


# --- Web search. Claude Code's built-in WebSearch is not available on Bedrock (the agent is never
#     offered it and answers from memory), so web_search is a custom tool. It calls Bedrock's own web
#     search, the same one the OpenAI deep research example uses: a Responses API request to a model on
#     the bedrock-mantle endpoint, which searches and replies with a summary and url_citation links. ---

TRACKING_PARAMS = {"msclkid", "gclid", "fbclid", "msockid", "click_product"}


def clean_url(url: str) -> str:
    """Drop ad-tracking query parameters that search results often carry, keeping ones the page needs."""
    parts = urlsplit(url)
    query_params = [(k, v) for k, v in parse_qsl(parts.query) if k not in TRACKING_PARAMS and not k.startswith("utm_")]
    return urlunsplit(parts._replace(query=urlencode(query_params)))


@dataclass
class BedrockWebSearch:
    """Runs Bedrock web search requests, and counts them: they are billed to the search model, so
    they don't appear in the Claude runs' total_cost_usd."""
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
            # No tool_choice="required": in October 2026 tests every such request timed out, while the
            # default "auto" plus the instruction above still searched each time
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
        """The search as an MCP tool the Claude agents can call."""
        @tool("web_search", "Search the web and get a short summary of the results with source URLs.",
              {"query": Annotated[str, "The search terms to use"]})
        async def web_search(args):
            try:
                text = await self.search(args["query"])
            except APIError as e:
                # The agent gets the error as the tool result, so it can try other terms or carry on
                print(f"   web_search {args['query']!r} failed: {e}")
                return {"content": [{"type": "text", "text": f"Web search failed: {e}"}], "is_error": True}
            return {"content": [{"type": "text", "text": text}]}
        return web_search


@tool("save_report", "Save a finished research report as a Markdown file and return the file path.",
      {"title": Annotated[str, "Short report title, e.g. 'Quiet Autumn Leaves Near Tokyo'."],
       "markdown": Annotated[str, "The full report in Markdown."]})
async def save_report(args):
    title, markdown = args["title"], args["markdown"]
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
    text = f"Saved to {path.relative_to(Path(__file__).parent.parent).as_posix()}"
    return {"content": [{"type": "text", "text": text}]}


# --- Agents: as in the workflow example, an agent here is a name, a system prompt and its tools,
#     turned into ClaudeAgentOptions for each run ---

@dataclass
class Agent:
    name: str
    system_prompt: str
    tools: list = field(default_factory=list)  # @tool functions, served by an in-process MCP server
    built_in_tools: list[str] = field(default_factory=list)  # Claude Code tools, e.g. WebFetch
    max_turns: int = 3


@dataclass
class Run:
    result: ResultMessage
    tool_calls: list[tuple[ToolUseBlock, str]]  # each call with the text it returned


def tool_name(sdk_tool) -> str:
    """The name the agent sees for an MCP tool: mcp__<server>__<tool>."""
    return f"mcp__{SERVER_NAME}__{sdk_tool.name}"


def result_text(block: ToolResultBlock) -> str:
    """A tool result's content is either a string or a list of content blocks."""
    if isinstance(block.content, list):
        return " ".join(part.get("text", "") for part in block.content)
    return block.content or "(no result)"


class StepTimeout(Exception):
    """An agent didn't finish its step in time."""


@dataclass
class Claude:
    """Where every run gets its model and credentials, and where every run is recorded for the cost
    summary and the clean-up."""
    model_id: str
    profile: str
    region: str
    runs: list[ResultMessage] = field(default_factory=list)
    sessions: set[str] = field(default_factory=set)  # recorded as each run starts, so timed-out ones too

    def options(self, agent: Agent) -> ClaudeAgentOptions:
        """Factory: the ClaudeAgentOptions for one agent, with all its tools pre-approved."""
        return ClaudeAgentOptions(
            model=self.model_id,
            system_prompt=agent.system_prompt,
            tools=agent.built_in_tools,  # only the built-in tools this agent needs; [] for most
            mcp_servers={SERVER_NAME: create_sdk_mcp_server(name=SERVER_NAME, tools=agent.tools)},
            allowed_tools=[tool_name(t) for t in agent.tools] + agent.built_in_tools,  # no one to ask
            max_turns=agent.max_turns,
            setting_sources=[],  # don't load ~/.claude or project settings and CLAUDE.md files into the agent
            env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": self.profile, "AWS_REGION": self.region,
                 # The CLI otherwise waits 10 minutes for a model request
                 "API_TIMEOUT_MS": str(MODEL_TIMEOUT * 1000), "CLAUDE_CODE_MAX_RETRIES": str(MODEL_RETRIES)},
        )

    async def run(self, agent: Agent, prompt: str) -> Run:
        """Run one agent on one prompt and collect its tool calls with their results, giving up after
        STEP_TIMEOUT seconds, or when a model request times out after its retries."""
        try:
            return await asyncio.wait_for(self.collect(agent, prompt), STEP_TIMEOUT)
        except TimeoutError:
            # wait_for cancels the query, which stops the CLI process
            raise StepTimeout(f"{agent.name} did not finish within {STEP_TIMEOUT}s") from None
        except ResultError as e:
            if e.terminal_reason == "api_error" and "timed out" in (e.result or ""):
                raise StepTimeout(f"{agent.name}'s model request timed out after {MODEL_TIMEOUT}s") from None
            raise

    async def collect(self, agent: Agent, prompt: str) -> Run:
        calls: list[ToolUseBlock] = []
        outputs: dict[str, str] = {}
        final = None
        async for message in query(prompt=prompt, options=self.options(agent)):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                self.sessions.add(message.data["session_id"])
            elif isinstance(message, AssistantMessage):
                calls += [block for block in message.content if isinstance(block, ToolUseBlock)]
            elif isinstance(message, UserMessage) and isinstance(message.content, list):
                outputs |= {b.tool_use_id: result_text(b) for b in message.content if isinstance(b, ToolResultBlock)}
            elif isinstance(message, ResultMessage):
                final = message
        self.runs.append(final)
        return Run(final, [(call, outputs.get(call.id, "(no result)")) for call in calls])

    async def ask_for(self, schema: type[BaseModel], agent: Agent, prompt: str) -> tuple[BaseModel, Run]:
        """Run an agent that answers through its submit_* tool, and return the typed object. The SDK
        validates the call against the schema, so the last call is always a valid one."""
        run = await self.run(agent, prompt)
        submit_names = {tool_name(t) for t in agent.tools if t.name.startswith("submit_")}
        submissions = [call for call, _ in run.tool_calls if call.name in submit_names]
        if not submissions:
            sys.exit(f"{agent.name} never submitted a {schema.__name__}. It said: {run.result.result}")
        return schema.model_validate(submissions[-1].input), run


def create_agents(web_search) -> dict[str, Agent]:
    """Factory: return the agents of the research pipeline, keyed by role."""
    date_note = f"Today is {today()} in Japan."
    return {
        "planner": Agent(
            "Planner",
            f"You are a travel research assistant. {date_note} Given a query, come up with "
            f"{HOW_MANY_SEARCHES} web searches that together best answer it. Give each search a "
            "different angle (for example forecast timing, lesser-known spots, or crowd and access tips) "
            "so they don't overlap, and keep each one within the area the query asks about. "
            "Answer only by calling submit_search_plan.",
            tools=[submit_search_plan],
        ),
        "searcher": Agent(
            "Searcher",
            f"{date_note} You research one search term. Always call web_search first, rather than answering "
            "from memory. Then write a concise summary of the results in under 120 words, with only facts "
            "useful to a traveler, noting which year any dates or forecasts refer to. End with the source "
            "URLs you relied on.",
            tools=[web_search],
            max_turns=4,  # a search, perhaps a second one, and the summary
        ),
        "writer": Agent(
            "Writer",
            f"{date_note} You write a cohesive travel research report from a query and search summaries. "
            "Use only facts from the summaries. Aim for about 400 words in Markdown with short sections, "
            "and end with a Sources section listing the URLs you relied on. "
            "Answer only by calling submit_report.",
            tools=[submit_report],
        ),
        "fact_checker": Agent(
            "Fact Checker",
            f"{date_note} You fact-check a travel research report against the search summaries it was "
            "written from. Flag claims that contradict the summaries or each other, claims no summary "
            "supports, outdated information, and places described with the wrong location or in the wrong "
            "section. The summaries can be wrong too, so verify the most specific facts, such as distances, "
            f"travel times and which place has which feature: use web_search to look them up, and "
            f"{WEB_FETCH} to read a source page. Only report real problems; minor wording is fine. "
            "Answer only by calling submit_fact_check.",
            tools=[web_search, submit_fact_check],
            built_in_tools=[WEB_FETCH],  # the one built-in tool that needs no local files or shell
            max_turns=10,  # room for a few lookups before it submits
        ),
        "publisher": Agent(
            "Publisher",
            "You save the report you are given with save_report using a clear title, "
            "then reply with where it was saved.",
            tools=[save_report],
        ),
    }


# --- The pipeline: plain Python code decides the order, agents do each step ---

async def plan_searches(claude: Claude, planner: Agent, research_query: str) -> SearchPlan:
    print(f"1. {planner.name}: planning searches...")
    plan, _ = await claude.ask_for(SearchPlan, planner, f"Query: {research_query}")
    for i, item in enumerate(plan.searches, 1):
        print(f"   {i}. {item.query!r} - {item.reason}")
    print()
    return plan


async def search(claude: Claude, searcher: Agent, item: SearchItem) -> str:
    """Research one search term and return the searcher's summary."""
    try:
        run = await claude.run(searcher, f"Search term: {item.query}\nReason for searching: {item.reason}")
    except StepTimeout as e:
        # One lost search shouldn't stop the report, so the writer gets the others
        print(f"   * {item.query!r}: {e}, skipped")
        return f"(No results for {item.query!r}: the search timed out.)"
    lookups = [call.input.get("query") for call, _ in run.tool_calls if call.name.endswith("__web_search")]
    print(f"   * {item.query!r}: {len(lookups)} web_search call(s): {lookups}")
    return run.result.result


async def run_searches(claude: Claude, searcher: Agent, web: BedrockWebSearch, plan: SearchPlan) -> list[str]:
    print(f"2. {searcher.name}: running {len(plan.searches)} searches in parallel...")
    summaries = await asyncio.gather(*(search(claude, searcher, item) for item in plan.searches))
    if web.calls == 0:
        # Searchers still reply when every search fails, and the writer would then write from memory
        sys.exit("Stopped: no web search succeeded, so there is nothing to base a report on.")
    print()
    for item, summary in zip(plan.searches, summaries):
        print(f"   --- Summary for {item.query!r} ---")
        print("   " + summary.strip().replace("\n", "\n   ") + "\n")
    return summaries


async def write_report(claude: Claude, writer: Agent, research_query: str, summaries: list[str]) -> ReportData:
    print(f"3. {writer.name}: writing the report...")
    report, _ = await claude.ask_for(ReportData, writer,
                                     f"Original query: {research_query}\nSummarized search results: {summaries}")
    print(f"   Summary: {report.short_summary}\n")
    return report


async def check_facts(claude: Claude, fact_checker: Agent, report: ReportData, summaries: list[str],
                      round_no: int, fixed: list[FactIssue]) -> FactCheck:
    print(f"4. {fact_checker.name} (round {round_no}/{MAX_REVIEW_ROUNDS}): checking the report...")
    prompt = f"Report:\n{report.markdown_report}\n\nSearch summaries: {summaries}"
    if fixed:
        # Without this, a later round can re-flag a fixed claim or undo a correction
        done = "\n".join(f"- {i.claim} -> {i.correction}" for i in fixed)
        prompt += f"\n\nCorrections already applied in earlier rounds (don't re-flag them unless still wrong):\n{done}"
    check, run = await claude.ask_for(FactCheck, fact_checker, prompt)
    # Show what was verified, so "0 issues" can be judged by what the checker actually looked up
    lookups = [(call.name, call.input) for call, _ in run.tool_calls if not call.name.endswith("__submit_fact_check")]
    print(f"   {len(check.issues)} issue(s) found, {len(lookups)} verification lookup(s)")
    for name, args in lookups:
        if name == WEB_FETCH:
            print(f"   - fetched:  {args.get('url')}")
        else:
            print(f"   - searched: {args.get('query')}")
    for i, issue in enumerate(check.issues, 1):
        print(f"   {i}. Claim:      {issue.claim}")
        print(f"      Problem:    {issue.problem}")
        print(f"      Correction: {issue.correction}")
    print()
    return check


async def revise_report(claude: Claude, writer: Agent, report: ReportData, check: FactCheck,
                        summaries: list[str]) -> ReportData:
    """Have the writer fix the fact checker's issues."""
    print(f"   {writer.name}: revising the report to fix {len(check.issues)} issue(s)...")
    issues = "\n".join(f"- {i.claim} -> {i.correction}" for i in check.issues)
    # Send the whole ReportData, so the writer keeps the follow-up questions and summarizes findings, not its edits
    revised, _ = await claude.ask_for(ReportData, writer,
                                      "Revise this report. Correct each flagged claim wherever it appears, "
                                      "without repeating information elsewhere, and keep everything else, "
                                      "including the follow-up questions. The short_summary must still summarize "
                                      "the findings, not describe your edits.\n\n"
                                      f"Issues:\n{issues}\n\nReport:\n{report.model_dump_json(indent=2)}\n\n"
                                      f"Search summaries: {summaries}")
    print(f"   Revised summary: {revised.short_summary}\n")
    return revised


async def review_report(claude: Claude, fact_checker: Agent, writer: Agent, report: ReportData,
                        summaries: list[str]) -> ReportData:
    """Check and revise until the fact checker finds no issues, for at most MAX_REVIEW_ROUNDS rounds."""
    fixed: list[FactIssue] = []
    for round_no in range(1, MAX_REVIEW_ROUNDS + 1):
        try:
            check = await check_facts(claude, fact_checker, report, summaries, round_no, fixed)
        except StepTimeout as e:
            # The report still exists, so publish it rather than lose the run
            print(f"   {e}. Publishing the report without this check.\n")
            return report
        if not check.issues:
            print(f"   Review passed in round {round_no}.\n")
            return report
        report = await revise_report(claude, writer, report, check, summaries)
        fixed += check.issues
    # The cap stops a checker that keeps finding something from looping (and costing) forever
    print(f"   Stopped after {MAX_REVIEW_ROUNDS} rounds; the last revision was not re-checked.\n")
    return report


async def publish_report(claude: Claude, publisher: Agent, report: ReportData) -> str:
    print(f"5. {publisher.name}: saving the report...")
    run = await claude.run(publisher, report.markdown_report)
    for call, output in run.tool_calls:
        print(f"   Called {call.name}(title={call.input.get('title')!r}, "
              f"markdown=<{len(call.input.get('markdown', ''))} chars>) -> {output}")
    print()
    return run.result.result


async def main(research_query: str):

    # 1. Debug Verification: Print environmental variables loaded by uv
    profile = os.environ.get("AWS_PROFILE", "Not Found")
    region = os.environ.get("AWS_REGION", "Not Found")
    model_id = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", "global.anthropic.claude-sonnet-5")
    search_model_id = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-5.6-luna")

    print("--- Environment Status ---")
    print(f"Active AWS Profile: {profile}")
    print(f"Active AWS Region:  {region}")
    print(f"Claude Model ID:    {model_id}")
    print(f"Search Model ID:    {search_model_id} (runs Bedrock web search)")
    print("--------------------------\n")

    # 2. The web search tool, and the pipeline's agents
    web = BedrockWebSearch(AsyncOpenAI(provider=bedrock(region=region), timeout=SEARCH_TIMEOUT, max_retries=1),
                           search_model_id)
    agents = create_agents(web.as_tool())
    claude = Claude(model_id, profile, region)

    print("--- Pipeline ---")
    print(f"1. {agents['planner'].name}:      query -> {SearchPlan.__name__} (via {submit_search_plan.name})")
    print(f"2. {agents['searcher'].name}:     each search -> summary, using web_search, in parallel")
    print(f"3. {agents['writer'].name}:       summaries -> {ReportData.__name__} (via {submit_report.name})")
    print(f"4. {agents['fact_checker'].name}: report -> {FactCheck.__name__} (via {submit_fact_check.name}), "
          f"using web_search and {WEB_FETCH};")
    print(f"                   while it finds issues, the Writer revises and it re-checks "
          f"(up to {MAX_REVIEW_ROUNDS} rounds)")
    print(f"5. {agents['publisher'].name}:    report -> Markdown file, using the save_report tool")
    print(f"Date given to agents: {today()} (Japan time)")
    print(f"web_search: Bedrock web search on {search_model_id}, external_web_access=False")
    print(f"Timeouts: web search {SEARCH_TIMEOUT}s (1 retry), model request {MODEL_TIMEOUT}s "
          f"({MODEL_RETRIES} retries), agent step {STEP_TIMEOUT}s\n")

    print("--- Query ---")
    print(f"{research_query}\n")

    print(f"Running on {model_id} via Amazon Bedrock...\n")
    try:
        plan = await plan_searches(claude, agents["planner"], research_query)
        summaries = await run_searches(claude, agents["searcher"], web, plan)
        report = await write_report(claude, agents["writer"], research_query, summaries)
        report = await review_report(claude, agents["fact_checker"], agents["writer"], report, summaries)
        await publish_report(claude, agents["publisher"], report)

        print("--- Report ---")
        print(report.markdown_report)
        print("\n--- Follow-up questions ---")
        for question in report.follow_up_questions:
            print(f"* {question}")

        # 3. Two bills: Claude reports its cost per run, and web search is billed to the search model
        print("\n--- Run Summary ---")
        print(f"Claude runs:  {len(claude.runs)}")
        print(f"Claude cost:  ${sum(run.total_cost_usd or 0 for run in claude.runs):.4f}")
        print(f"Web searches: {web.calls} on {search_model_id}, "
              f"{web.input_tokens} in / {web.output_tokens} out tokens (not in the Claude cost)")
    finally:
        # 4. Clean up, even when a step failed: none of these sessions is ever resumed, so remove
        #    their transcripts
        for session_id in claude.sessions:
            delete_session(session_id)
        print(f"\n--- {len(claude.sessions)} sessions deleted (delete_session) ---")

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
