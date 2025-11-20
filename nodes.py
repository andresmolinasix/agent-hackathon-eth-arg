import json
import os
import pathlib
from types import SimpleNamespace
from typing import Optional
from urllib import parse, request

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pocketflow import Node, Flow

load_dotenv()


_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def call_llm(prompt: str, expect_yaml: bool = False) -> str:
    """Send the prompt to OpenAI and return the raw text response."""
    completion = _client.responses.create(
        model="gpt-4.1-mini",  # or whichever model you prefer
        input=[{"role": "user", "content": prompt}],
    )

    parts = []
    for item in getattr(completion, "output", []) or []:
        if getattr(item, "type", "") == "message":
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    parts.append(content.text)

    if not parts and getattr(completion, "output_text", None):
        parts.append("".join(completion.output_text))

    text = "".join(parts).strip()
    if expect_yaml and "```yaml" not in text:
        # Gracefully degrade to a default YAML block so downstream nodes keep working.
        fallback = (
            "```yaml\n"
            "thinking: |\n"
            "    Model response did not include YAML. Original text was returned instead.\n"
            "action: answer\n"
            "reason: Defaulting to answering because no structured output was provided.\n"
            "parameters:\n"
            "    search_query: \"\"\n"
            "    report_content: \"\"\n"
            "    target_path: \"\"\n"
            "    answer_outline: \"\"\n"
            "    finish_after_write: false\n"
            "```"
        )
        return fallback
    return text


class GoogleSearchAPI:
    """Minimal Google Custom Search API client."""

    SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.environ.get("GOOGLE_CSE_ID")
        if not self.api_key:
            raise RuntimeError("Missing Google API key. Set GOOGLE_API_KEY in your environment.")
        if not self.search_engine_id:
            raise RuntimeError("Missing Google Search Engine ID. Set GOOGLE_CSE_ID in your environment.")

    def search(self, params: dict):
        query = params.get("query")
        if not query:
            raise ValueError("Search query cannot be empty.")

        num_results = max(1, min(int(params.get("num_results", 3)), 10))
        language = params.get("language")
        request_params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": num_results,
        }
        if language:
            request_params["lr"] = f"lang_{language}"

        url = f"{self.SEARCH_ENDPOINT}?{parse.urlencode(request_params)}"
        with request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Google Search API error: HTTP {resp.status}")
            data = json.load(resp)
        items = data.get("items", [])

        results = []
        for item in items:
            results.append(
                SimpleNamespace(
                    title=item.get("title", "No title"),
                    snippet=item.get("snippet", ""),
                    link=item.get("link", ""),
                )
            )
        return results

class DecideAction(Node):
    def prep(self, shared):
        # Think of "shared" as a big notebook that everyone can read and write in
        # It's where we store everything our agent knows

        # Look for any previous research we've done (if we haven't searched yet, just note that)
        context = shared.get("context", "No previous search")

        # Get the question we're trying to answer
        question = shared["question"]

        # Return both pieces of information for the next step
        return question, context
    
    def exec(self, inputs):
        # This is where the magic happens - the LLM "thinks" about what to do
        question, context = inputs

        # We ask the LLM to decide what to do next with this prompt:
        prompt = f"""
        ### CONTEXT
        You are a research assistant that can search the web, draft answers, and save reports to disk.
        Question: {question}
        Previous Research: {context}

        ### ACTION SPACE
        [1] search
        Description: Look up more information on the web.
        Parameters: search_query (required) — a concrete search phrase.

        [2] write
        Description: Save finalized content into a file (use only when text is ready to store).
        Parameters:
            - report_content (required): plain text to write.
            - target_path (optional): output file path, defaults to report.txt.
            - finish_after_write (optional bool): true if the workflow can stop after writing.

        [3] answer
        Description: Provide the best possible answer with the current research. Set target_path if the user requested a file so we can write after answering.
        Parameters:
            - answer_outline (optional): guidance for the AnswerQuestion node.
            - target_path (optional): file path to save the final answer after writing.

        ## NEXT ACTION
        Decide the next action based on the context and available actions.
        Return your response in this format:

        ```yaml
        thinking: |
            <your step-by-step reasoning process>
        action: search | write | answer
        reason: <why you chose this action>
        parameters:
            search_query: ""
            report_content: ""
            target_path: ""
            answer_outline: ""
            finish_after_write: false
        ```"""

        # Call the LLM to make a decision
        response = call_llm(prompt, expect_yaml=True)

        # Pull out just the organized information part from the LLM's answer
        # (This is like finding just the recipe part of a cooking video)
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()

        # Some model generations forget to quote the search query and produce invalid YAML.
        # Force-quote the value to avoid parser errors.
        fixed_lines = []
        for line in yaml_str.splitlines():
            stripped = line.strip()
            if stripped.startswith("search_query:") and not stripped.startswith("search_query: |"):
                key, _, value = line.partition(":")
                value = value.strip()
                if value and value[0] not in ("'", '"'):
                    line = f"{key}: {json.dumps(value)}"
            fixed_lines.append(line)
        yaml_str = "\n".join(fixed_lines)

        try:
            decision = yaml.safe_load(yaml_str) or {}  # Convert the text into a format our program can use
        except yaml.YAMLError as exc:
            decision = {
                "thinking": f"Failed to parse YAML decision. Raw content: {yaml_str}",
                "action": "answer",
                "reason": f"YAML parse error: {exc}. Defaulting to answer.",
                "parameters": {
                    "search_query": "",
                    "report_content": "",
                    "target_path": "",
                    "answer_outline": "",
                    "finish_after_write": False,
                },
            }

        params = decision.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        # Backwards compatibility for earlier schema versions.
        for legacy_key in ("search_query", "report_content", "target_path", "answer_outline", "finish_after_write"):
            if legacy_key in decision and legacy_key not in params:
                params[legacy_key] = decision[legacy_key]
        decision["parameters"] = params

        return decision
    
    def post(self, shared, prep_res, exec_res):
        question, _ = prep_res
        decision = exec_res
        params = decision.get("parameters", {}) or {}

        if decision["action"] == "search":
            shared["search_query"] = params.get("search_query") or question

        target_path = params.get("target_path")
        if target_path:
            shared["target_path"] = target_path

        answer_outline = params.get("answer_outline")
        if answer_outline:
            shared["answer_outline"] = answer_outline

        report_content = params.get("report_content")
        if report_content:
            shared["report_content"] = report_content

        finish_after_write = params.get("finish_after_write")
        if finish_after_write is not None:
            shared["finish_after_write"] = bool(finish_after_write)

        if decision["action"] == "write":
            # If no explicit report content was given, fall back to the last answer.
            if not shared.get("report_content") and shared.get("answer"):
                shared["report_content"] = shared["answer"]

        if decision["action"] == "answer" and "answer_outline" not in params:
            shared.setdefault("answer_outline", "")

        return decision["action"]



class SearchWeb(Node):
    def prep(self, shared):
        # Simply get the search query we saved earlier
        return shared["search_query"]
    
    def exec(self, search_query):
        # This is where we'd connect to Google to search the internet
        search_client = GoogleSearchAPI()

        # Set search parameters
        search_params = {
            "query": search_query,
            "num_results": 3,
            "language": "en"
        }

        # Make the API request to Google
        try:
            results = search_client.search(search_params)
        except Exception as exc:
            return f"Failed to fetch search results: {exc}"

        # Format the results into readable text
        formatted_results = f"Results for: {search_query}\n"

        # Process each search result
        for result in results:
            # Extract the title and snippet from each result
            formatted_results += f"- {result.title}: {result.snippet}\n"

        return formatted_results
    
    def post(self, shared, prep_res, exec_res):
        # Store the search results in our shared whiteboard
        previous = shared.get("context", "")
        shared["context"] = previous + "\n\nSEARCH: " + shared["search_query"] + "\nRESULTS: " + exec_res

        # Always go back to the decision node after searching
        return "decide" 

class AnswerQuestion(Node):
    def prep(self, shared):
        # Get both the original question and all the research we've done
        return shared["question"], shared.get("context", "")
    
    def exec(self, inputs):
        question, context = inputs
        # Ask the LLM to create a helpful answer based on our research
        prompt = f"""
        ### CONTEXT
        Based on the following information, answer the question.
        Question: {question}
        Research: {context}

        ## YOUR ANSWER:
        Provide a comprehensive answer using the research results.
        """
        return call_llm(prompt)
    
    def post(self, shared, prep_res, exec_res):
        # Save the answer in our shared whiteboard
        shared["answer"] = exec_res
        if not shared.get("report_content"):
            shared["report_content"] = exec_res

        if shared.get("target_path"):
            # Ensure we wrap up once the file is written unless explicitly told otherwise.
            shared.setdefault("finish_after_write", True)
            return "write"

        # We're done! No need to continue the flow.
        return "done"

class WriteFile(Node):
    def prep(self, shared):
        content = shared.get("report_content") or shared.get("answer", "")
        path = shared.get("target_path", "report.txt")
        return content, path

    def exec(self, inputs):
        content, path = inputs
        if not content:
            return "Nothing to write because no report content was provided."
        pathlib.Path(path).write_text(content, encoding="utf-8")
        return f"Saved report to {path}"

    def post(self, shared, prep_res, exec_res):
        shared.setdefault("context", "")
        shared["context"] += f"\n\nWRITE: {exec_res}"
        finish = shared.pop("finish_after_write", False)
        shared.pop("target_path", None)
        if finish:
            return "done"
        return "decide"          # route back to DecideAction

