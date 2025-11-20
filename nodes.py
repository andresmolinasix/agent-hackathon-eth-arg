import json
import os
import pathlib
import re
import time
from types import SimpleNamespace
from typing import Optional
from urllib import parse, request

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pocketflow import Node, Flow

TIMEOUT_SECONDS = 10

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
            "    price_token: {{}}\n"
            "    volume_filters: {{}}\n"
            "    yield_filters: {{}}\n"
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

class DeFiLlamaClient:
    """Lightweight client for DeFiLlama price, yield, and volume endpoints."""

    PRICE_BASE = "https://coins.llama.fi"
    YIELD_BASE = "https://yields.llama.fi"
    VOLUME_BASE = "https://api.llama.fi"

    def _fetch_json(self, url: str):
        with request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                raise RuntimeError(f"DeFiLlama API error: HTTP {resp.status}")
            return json.load(resp)

    def get_current_price(self, coin_id: str):
        if not coin_id:
            raise ValueError("Coin id is required for price lookup.")
        url = f"{self.PRICE_BASE}/prices/current/{coin_id}"
        return self._fetch_json(url)

    def get_historical_price(self, coin_id: str, timestamp: int):
        if not coin_id or not timestamp:
            raise ValueError("Coin id and timestamp are required for historical price.")
        url = f"{self.PRICE_BASE}/prices/historical/{timestamp}/{coin_id}"
        return self._fetch_json(url)

    def get_yields(self, filters: Optional[dict] = None):
        filters = filters or {}
        query = parse.urlencode({k: v for k, v in filters.items() if v})
        url = f"{self.YIELD_BASE}/pools"
        if query:
            url = f"{url}?{query}"
        return self._fetch_json(url)

    def get_volumes(self, filters: Optional[dict] = None):
        filters = filters or {}
        # /summary/dexvolumes with optional ?dex={dex}&chain={chain}
        query = parse.urlencode({k: v for k, v in filters.items() if v})
        url = f"{self.VOLUME_BASE}/summary/dexvolumes"
        if query:
            url = f"{url}?{query}"
        return self._fetch_json(url)

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

        # Fast-path: if the question clearly asks for a token price, bypass the LLM and go straight to fetch_price.
        def parse_price_intent(q: str):
            text = q.lower()
            if "price" not in text and "precio" not in text:
                return None
            # Extract chain:address or coingecko:symbol or plain chain name
            m = re.search(r"([a-z0-9_-]+):0x[0-9a-f]{40}", q, re.IGNORECASE)
            chain = address = coin_id = symbol = None
            if m:
                chain = m.group(1)
                address = re.search(r":(0x[0-9a-f]{40})", m.group(0), re.IGNORECASE).group(1)
            else:
                cg = re.search(r"coingecko:([a-z0-9-]+)", q, re.IGNORECASE)
                if cg:
                    coin_id = f"coingecko:{cg.group(1)}"
                else:
                    # Fallback: treat a bare chain name as coingecko:{chain}
                    bare_chain = None
                    if "ethereum" in text:
                        bare_chain = "ethereum"
                    elif "bitcoin" in text or "btc" in text:
                        bare_chain = "bitcoin"
                    if bare_chain:
                        coin_id = f"coingecko:{bare_chain}"
                        chain = bare_chain
            sym = re.search(r"\(([A-Za-z0-9]+)\)", q)
            symbol = sym.group(1) if sym else None

            draw_ts = None
            # Explicit timestamp
            match = re.search(r"drawdown_timestamp\\s*=\\s*(\\d{9,})", text)
            if match:
                try:
                    draw_ts = int(match.group(1))
                except ValueError:
                    draw_ts = None
            # Comparative phrasing
            comparative = any(kw in text for kw in ["compare", "versus", "vs ", "vs.", "change", "difference", "week ago", "last week", "1 week", "one week"])
            if comparative and not draw_ts:
                draw_ts = int(time.time()) - 7 * 24 * 3600

            if not any([coin_id, chain and address, chain]):
                return None
            params = {
                "price_token": {"chain": chain, "address": address, "id": coin_id, "symbol": symbol},
                "drawdown_timestamp": draw_ts,
            }
            return {
                "thinking": "Direct price request detected; bypassing planner.",
                "action": "fetch_price",
                "reason": "User explicitly asked for a token price.",
                "parameters": params,
                "force_answer_after_price": True,
            }

        parsed = parse_price_intent(question)
        if parsed:
            return parsed

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

        [3] fetch_price
        Description: Get current (and optional historical) token price from DeFiLlama.
        Parameters:
            - price_token: {{chain, address, symbol}}
            - drawdown_timestamp: optional unix timestamp for historical comparison.

        [4] fetch_yields
        Description: Fetch yields for the token/chain (lending, staking, lp) from DeFiLlama.
        Parameters:
            - yield_filters: {{chain, symbol, project, category}}

        [5] fetch_volume
        Description: Fetch DEX volume summary for a chain or specific DEX.
        Parameters:
            - volume_filters: {{chain, dex}}

        [6] assess_risk
        Description: Compute risk flags using prices, yields, and volume.
        Parameters:
            - preferences: {{drawdown_threshold, utilization_threshold, turnover_min}}

        [7] answer
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
        action: search | write | fetch_price | fetch_yields | fetch_volume | assess_risk | answer
        reason: <why you chose this action>
        parameters:
            search_query: ""
            report_content: ""
            target_path: ""
            answer_outline: ""
            finish_after_write: false
            price_token: {{}}
            drawdown_timestamp: null
            yield_filters: {{}}
            volume_filters: {{}}
            preferences: {{}}
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
        for legacy_key in (
            "search_query",
            "report_content",
            "target_path",
            "answer_outline",
            "finish_after_write",
            "price_token",
            "drawdown_timestamp",
            "yield_filters",
            "volume_filters",
            "preferences",
        ):
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

        price_token = params.get("price_token")
        if price_token:
            shared["price_token"] = price_token
        draw_ts = params.get("drawdown_timestamp")
        if draw_ts:
            shared["drawdown_timestamp"] = draw_ts
        else:
            # Try to infer a week-ago comparison from the question text.
            question_text = str(shared.get("question", "")).lower()
            if "week ago" in question_text or "1 week" in question_text or "one week" in question_text:
                shared["drawdown_timestamp"] = int(time.time()) - 7 * 24 * 3600
            else:
                # Attempt to extract a numeric timestamp embedded in the question if present.
                match = re.search(r"drawdown_timestamp\s*=\s*(\d{9,})", question_text)
                if match:
                    try:
                        shared["drawdown_timestamp"] = int(match.group(1))
                    except ValueError:
                        pass

        yield_filters = params.get("yield_filters")
        if yield_filters:
            shared["yield_filters"] = yield_filters
        volume_filters = params.get("volume_filters")
        if volume_filters:
            shared["volume_filters"] = volume_filters
        preferences = params.get("preferences")
        if preferences:
            shared["preferences"] = preferences

        # If the planner shortcut requested an immediate answer after price fetch, keep that flag.
        if decision.get("force_answer_after_price"):
            shared["force_answer_after_price"] = True

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
        return shared["question"], shared.get("context", ""), shared.get("prices", {})
    
    def exec(self, inputs):
        question, context, prices = inputs
        # If we already have structured price data, respond deterministically from it to avoid hallucinations.
        if prices:
            meta = next(iter(prices.values()))
            coin_id = meta.get("coin_id")
            current_coins = meta.get("current", {}).get("coins", {})
            if not coin_id and current_coins:
                coin_id = next(iter(current_coins.keys()))
            cur_entry = current_coins.get(coin_id, {}) if coin_id else {}
            curr_price = cur_entry.get("price")
            curr_ts = cur_entry.get("timestamp")
            hist_price = hist_ts = None
            if meta.get("history"):
                coins_hist = meta["history"].get("coins", {})
                hist_entry = coins_hist.get(coin_id, {}) if coin_id else {}
                hist_price = hist_entry.get("price")
                hist_ts = hist_entry.get("timestamp")
            drawdown = meta.get("drawdown")
            parts = []
            parts.append(f"Current price: {curr_price} (timestamp: {curr_ts})")
            if hist_price is not None:
                parts.append(f"Historical price: {hist_price} (timestamp: {hist_ts})")
            if drawdown is not None:
                parts.append(f"Drawdown over period: {drawdown:.2%}")
            return "\n".join(parts)

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


class FetchPrice(Node):
    def prep(self, shared):
        token = shared.get("price_token") or {}
        draw_ts = shared.get("drawdown_timestamp")
        return token, draw_ts

    def exec(self, inputs):
        token, draw_ts = inputs
        client = DeFiLlamaClient()
        chain = token.get("chain")
        address = token.get("address")
        coin_id = token.get("id")
        symbol = token.get("symbol", address or chain or coin_id or "unknown")

        # Build the DeFiLlama coin id. Accept explicit id, chain+address, or coingecko:{symbol} fallback.
        if coin_id:
            coin_key = coin_id
        elif chain and address:
            coin_key = f"{chain}:{address}"
        elif chain and not address:
            # Allow native coin lookup via coingecko namespace (e.g., coingecko:ethereum)
            coin_key = f"coingecko:{chain.lower()}"
        else:
            return "Missing coin identifier for price lookup."
        try:
            current = client.get_current_price(coin_key)
            history = None
            drawdown = None
            if draw_ts:
                history = client.get_historical_price(coin_key, int(draw_ts))
                try:
                    prices = history.get("coins", {})
                    past_price = prices.get(coin_key, {}).get("price")
                    curr_price = current.get("coins", {}).get(coin_key, {}).get("price")
                    if past_price and curr_price:
                        drawdown = (curr_price - past_price) / past_price
                except Exception:
                    drawdown = None
            return {
                "symbol": symbol,
                "chain": chain,
                "address": address,
                "coin_id": coin_key,
                "current": current,
                "history": history,
                "drawdown": drawdown,
            }
        except Exception as exc:
            return f"Failed to fetch price: {exc}"

    def post(self, shared, prep_res, exec_res):
        shared.setdefault("prices", {})
        if isinstance(exec_res, str):
            shared["context"] = shared.get("context", "") + f"\n\nPRICE ERROR: {exec_res}"
        else:
            symbol = exec_res.get("symbol")
            shared["prices"][symbol] = exec_res
            shared["context"] = shared.get("context", "") + f"\n\nPRICE: {symbol} {exec_res.get('current')}"
        if shared.get("force_answer_after_price"):
            return "answer"
        return "decide"


class FetchYields(Node):
    def prep(self, shared):
        filters = shared.get("yield_filters", {})
        token = shared.get("price_token", {})
        # Prefer explicit filters; fall back to token info for symbol/chain hints.
        filters = {**({"chain": token.get("chain")} if token else {}), **filters}
        return filters

    def exec(self, filters):
        client = DeFiLlamaClient()
        try:
            raw = client.get_yields(filters)
            pools = raw.get("data") or raw.get("pools") or []
            results = []
            for pool in pools:
                results.append(
                    {
                        "apyBase": pool.get("apyBase"),
                        "apyReward": pool.get("apyReward"),
                        "tvlUsd": pool.get("tvlUsd"),
                        "tvlBorrow": pool.get("tvlBorrow"),
                        "tvlSupply": pool.get("tvlSupply"),
                        "project": pool.get("project"),
                        "chain": pool.get("chain"),
                        "symbol": pool.get("symbol"),
                        "pool": pool.get("pool"),
                        "category": pool.get("ilRisk") or pool.get("category"),
                    }
                )
            return results
        except Exception as exc:
            return f"Failed to fetch yields: {exc}"

    def post(self, shared, prep_res, exec_res):
        if isinstance(exec_res, str):
            shared["context"] = shared.get("context", "") + f"\n\nYIELD ERROR: {exec_res}"
        else:
            shared["yields"] = exec_res
            shared["context"] = shared.get("context", "") + f"\n\nYIELDS: fetched {len(exec_res)} pools"
        return "decide"


class FetchVolume(Node):
    def prep(self, shared):
        return shared.get("volume_filters", {})

    def exec(self, filters):
        client = DeFiLlamaClient()
        try:
            raw = client.get_volumes(filters)
            # Keep a concise snapshot if possible.
            if isinstance(raw, dict):
                return raw
            return {"data": raw}
        except Exception as exc:
            return f"Failed to fetch volumes: {exc}"

    def post(self, shared, prep_res, exec_res):
        if isinstance(exec_res, str):
            shared["context"] = shared.get("context", "") + f"\n\nVOLUME ERROR: {exec_res}"
        else:
            shared["volumes"] = exec_res
            shared["context"] = shared.get("context", "") + "\n\nVOLUMES: data fetched"
        return "decide"


class AssessRisk(Node):
    def prep(self, shared):
        prefs = shared.get("preferences", {}) or {}
        return (
            shared.get("prices", {}),
            shared.get("yields", []),
            shared.get("volumes", {}),
            {
                "drawdown_threshold": prefs.get("drawdown_threshold", 0.2),
                "utilization_threshold": prefs.get("utilization_threshold", 0.9),
                "turnover_min": prefs.get("turnover_min", 0.05),
            },
        )

    def exec(self, inputs):
        prices, yields, volumes, prefs = inputs
        notes = []
        result = {"staking": None, "lending": None, "lp": None}

        # Staking: drawdown
        drawdown_flag = None
        for meta in prices.values():
            dd = meta.get("drawdown")
            if dd is not None:
                drawdown_flag = dd <= -abs(prefs["drawdown_threshold"])
                notes.append(f"Drawdown {dd:.2%} vs threshold {prefs['drawdown_threshold']:.0%}")
                break
        result["staking"] = drawdown_flag

        # Lending: utilization
        util_flag = None
        for pool in yields or []:
            borrow = pool.get("tvlBorrow")
            supply = pool.get("tvlSupply") or pool.get("tvlUsd")
            if borrow and supply:
                util = borrow / supply if supply else 0
                util_flag = util >= prefs["utilization_threshold"]
                notes.append(f"Utilization {util:.2%} vs {prefs['utilization_threshold']:.0%}")
                break
        result["lending"] = util_flag

        # LP: turnover
        lp_flag = None
        turnover_min = prefs["turnover_min"]
        # Try to estimate turnover using volumes (24h) and yields TVL
        vol_24h = None
        if isinstance(volumes, dict):
            vol_24h = volumes.get("total24h") or volumes.get("volume24h")
            if not vol_24h:
                for v in volumes.values():
                    if isinstance(v, dict) and v.get("volume24h"):
                        vol_24h = v.get("volume24h")
                        break
        tvl = None
        for pool in yields or []:
            if pool.get("tvlUsd"):
                tvl = pool["tvlUsd"]
                break
        if vol_24h and tvl:
            turnover = vol_24h / tvl if tvl else 0
            lp_flag = turnover < turnover_min
            notes.append(f"Turnover {turnover:.2f} vs min {turnover_min}")
        result["lp"] = lp_flag

        missing = []
        if not prices:
            missing.append("prices")
        if yields is None:
            missing.append("yields")
        if not volumes:
            missing.append("volumes")

        recommendation = []
        if drawdown_flag:
            recommendation.append("Staking risk: price drawdown exceeds comfort.")
        if util_flag:
            recommendation.append("Lending risk: high utilization; liquidation/systemic risk elevated.")
        if lp_flag:
            recommendation.append("LP risk: low turnover relative to TVL; fees may not offset IL.")
        if not recommendation:
            recommendation.append("No major risks detected given available data.")

        if missing:
            recommendation.append(f"Missing data: {', '.join(missing)}.")

        return {"flags": result, "notes": notes, "recommendation": " ".join(recommendation)}

    def post(self, shared, prep_res, exec_res):
        shared["risk"] = exec_res
        shared["answer"] = exec_res.get("recommendation", "")
        shared.setdefault("context", "")
        shared["context"] += "\n\nRISK:\n" + shared["answer"]
        return "answer"
