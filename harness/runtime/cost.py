"""Token usage, cost and step counts of agent runs.

    python harness/runtime/cost.py <runs_root>    # recompute them into every run.json

Prices come from LiteLLM's price table at a pinned revision. A model is looked up
by its normalised name: lower case, no provider path, no serving suffix (`-fp8`,
`-sglang`, ...). `stirrup` models use OpenRouter's rates first, the CLIs' models
their provider's. An unlisted model keeps its usage and gets a null `cost_usd`.

`steps` counts model responses and `tool_calls` the tool invocations in them,
both read from the session transcript.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

from agents import agent

PRICES_URL = ("https://raw.githubusercontent.com/BerriAI/litellm/"
              "b79fc9f1b0249cb533391dd7c941082f0cd6ddb1/model_prices_and_context_window.json")
PRICES_CACHE = Path.home() / ".cache" / "4dcodebench" / "model_prices_and_context_window.json"
# suffixes of served model names that the price table does not carry
SERVING_SUFFIXES = ("-sglang", "-vllm", "-fp8", "-nvfp4", "-awq", "-int4", "-int8")
PRICE_FIELDS = ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost",
                "cache_creation_input_token_cost")
# Stirrup usage counts each request's prefix shared with the previous one as cached;
# cost bills this fraction of those tokens at the cache-read rate.
CACHE_HIT_RATE = 0.95


def _lines(path: Path) -> list[dict]:
    # JSON lines of a transcript; unparsable lines (a truncated last line) are skipped
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            try:
                found.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return found


def _stirrup_usage(session: Path) -> dict | None:
    # sums `step` lines, whose token counters are cumulative per file; the input shared
    # with the previous request counts as cached
    inputs, completion = [], 0
    for transcript in sorted(session.rglob("[0-9]*.jsonl")):
        last_in = last_out = 0
        for item in _lines(transcript):
            if item.get("type") != "step":
                continue
            inputs.append(item["input_tokens"] - last_in)
            completion += item["output_tokens"] - last_out
            last_in, last_out = item["input_tokens"], item["output_tokens"]
    if not inputs:
        return None
    cached = sum(min(previous, current) for previous, current in zip(inputs, inputs[1:]))
    return {
        "input_tokens": sum(inputs) - cached,
        "cached_input_tokens": cached,
        "cache_write_tokens": 0,
        "output_tokens": completion,
        "reasoning_output_tokens": 0,
    }


def _accumulated_usage(source: Path, family: str) -> dict | None:
    """Sum token counts over a claude or codex session's request records, subagents included."""

    totals = dict.fromkeys(("input_tokens", "cached_input_tokens", "cache_write_tokens",
                            "output_tokens", "reasoning_output_tokens"), 0)
    files = sorted(source.rglob("*.jsonl")) if source.is_dir() else [source] if source.is_file() else []
    counted, seen = False, set()
    for transcript in files:
        for item in _lines(transcript):
            if family == "claude":
                # one line per content block; usage repeats per message id, counted once
                message = item.get("message") if isinstance(item.get("message"), dict) else {}
                raw = message.get("usage")
                if not raw or message.get("id") in seen:
                    continue
                seen.add(message.get("id"))
                totals["input_tokens"] += raw.get("input_tokens", 0)
                totals["cached_input_tokens"] += raw.get("cache_read_input_tokens", 0)
                totals["cache_write_tokens"] += raw.get("cache_creation_input_tokens", 0)
                totals["output_tokens"] += raw.get("output_tokens", 0)
                totals["reasoning_output_tokens"] += (raw.get("output_tokens_details") or {}).get("thinking_tokens", 0)
            elif family == "codex":
                payload = item.get("payload")
                if item.get("type") != "event_msg" or not isinstance(payload, dict) \
                        or payload.get("type") != "token_count":
                    continue
                raw = (payload.get("info") or {}).get("last_token_usage")
                if not raw:
                    continue
                totals["input_tokens"] += raw.get("input_tokens", 0) - raw.get("cached_input_tokens", 0)
                totals["cached_input_tokens"] += raw.get("cached_input_tokens", 0)
                totals["output_tokens"] += raw.get("output_tokens", 0)
                totals["reasoning_output_tokens"] += raw.get("reasoning_output_tokens", 0)
            else:
                return None
            counted = True
    return totals if counted else None


def summarize_cost(path: Path, family: str, model: str) -> dict | None:
    """Return usage and cost of one run from its agent.log, falling back to the session."""

    if family == "stirrup":   # read from the session transcript
        usage = _stirrup_usage(path.parents[1] / "session")
        if usage is None:
            return None
    elif family == "antigravity":
        payloads = _lines(path)
        payload = next((item for item in reversed(payloads) if "usage" in item), None)
        if payload is None:
            return None
        raw = payload["usage"]
        usage = {
            "input_tokens": raw["input_tokens"],
            "cached_input_tokens": raw.get("cache_read_tokens", 0),
            "cache_write_tokens": 0,
            "output_tokens": raw["output_tokens"],
            "reasoning_output_tokens": raw.get("thinking_tokens", 0),
        }
    elif family == "claude":
        # summed from the session transcripts; the agent.log summary line is the fallback
        usage = _accumulated_usage(path.parents[1] / "session", family)
        payload = next((item for item in reversed(_lines(path)) if "usage" in item), None)
        if usage is None and payload is None:
            return None
        if usage is None:
            raw = payload["usage"]
            usage = {
                "input_tokens": raw["input_tokens"],
                "cached_input_tokens": raw["cache_read_input_tokens"],
                "cache_write_tokens": raw["cache_creation_input_tokens"],
                "output_tokens": raw["output_tokens"],
                "reasoning_output_tokens": raw.get("output_tokens_details", {}).get("thinking_tokens", 0),
            }
    elif family == "codex":
        payloads = _lines(path)
        payload = next((item for item in reversed(payloads) if item.get("type") == "turn.completed"), None)
        if payload is None:
            usage = _accumulated_usage(path.parents[1] / "session", family)
            return None if usage is None else _price(usage, family, model)
        raw = payload["usage"]
        usage = {
            "input_tokens": raw["input_tokens"] - raw["cached_input_tokens"],
            "cached_input_tokens": raw["cached_input_tokens"],
            "cache_write_tokens": 0,
            "output_tokens": raw["output_tokens"],
            "reasoning_output_tokens": raw["reasoning_output_tokens"],
        }
    else:
        raise ValueError(f"unsupported agent family {family!r}")

    return _price(usage, family, model)


def _table() -> dict:
    """Return the pinned price table, downloading it once into the user's cache."""

    if not PRICES_CACHE.is_file():
        PRICES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(PRICES_URL, timeout=60) as response:
            data = response.read()
        partial = PRICES_CACHE.with_suffix(".part")
        partial.write_bytes(data)
        partial.replace(PRICES_CACHE)
    return json.loads(PRICES_CACHE.read_text(encoding="utf-8"))


def normalize(model: str) -> str:
    """Return `model` lower-cased, without provider path or serving suffixes."""

    name = model.lower().rsplit("/", 1)[-1]
    while name.endswith(SERVING_SUFFIXES):
        name = name[:name.rindex("-")]
    return name


def price_entry(table: dict, model: str, family: str) -> tuple[str, dict] | None:
    """Return the table's key and entry for `model`, or None when it is not listed."""

    name = normalize(model)
    exact = [key for key in table if key.lower() == name]
    routed = sorted(key for key in table
                    if key.lower().startswith("openrouter/") and key.lower().rsplit("/", 1)[-1] == name)
    ordered = routed + exact if family == "stirrup" else exact + routed
    if not ordered:
        # listed only under other providers: used when they all charge the same
        others = sorted(key for key in table if key.lower().rsplit("/", 1)[-1] == name)
        rates = {json.dumps({field: table[key].get(field) for field in PRICE_FIELDS}) for key in others}
        ordered = others if len(rates) == 1 else []
    return (ordered[0], table[ordered[0]]) if ordered else None


def _price(usage: dict, family: str, model: str) -> dict:
    """Return usage and its cost in USD; the cost is None for a model without a price."""

    hits = round(usage["cached_input_tokens"] * CACHE_HIT_RATE) if family == "stirrup" else usage["cached_input_tokens"]
    misses = usage["input_tokens"] + usage["cached_input_tokens"] - hits
    try:
        found = price_entry(_table(), model, family)
    except OSError:          # no network and no cached table: usage without cost
        found = None
    entry = found[1] if found else {}
    rate, output = entry.get("input_cost_per_token"), entry.get("output_cost_per_token")
    if rate is None or output is None:
        return {"usage": usage, "cost_usd": None}
    # without a cache price, cached and written tokens bill at the input rate
    cached = entry["cache_read_input_token_cost"] if entry.get("cache_read_input_token_cost") is not None else rate
    written = entry["cache_creation_input_token_cost"] if entry.get("cache_creation_input_token_cost") is not None else rate
    cost = (misses * rate + hits * cached + usage["cache_write_tokens"] * written
            + usage["output_tokens"] * output)
    return {"usage": usage, "cost_usd": round(cost, 6)}


def _claude_steps(session_root: Path) -> tuple[int, int] | None:
    # one `assistant` line per content block; a model call is one message id
    transcripts = sorted(session_root.rglob("*.jsonl"))
    if not transcripts:
        return None
    calls, tools = set(), 0
    for transcript in transcripts:
        for item in _lines(transcript):
            if item.get("type") != "assistant":
                continue
            message = item.get("message") or {}
            calls.add(message.get("id") or item.get("uuid"))
            tools += sum(1 for block in message.get("content") or [] if block.get("type") == "tool_use")
    return len(calls), tools


def _codex_steps(session: Path) -> tuple[int, int] | None:
    # one `token_count` event per model response; tool calls are response items
    transcripts = sorted(session.rglob("rollout-*.jsonl"))
    if not transcripts:
        return None
    calls = tools = 0
    for transcript in transcripts:
        for item in _lines(transcript):
            payload = item.get("payload") or {}
            kind = payload.get("type") if isinstance(payload, dict) else None
            if item.get("type") == "event_msg" and kind == "token_count":
                calls += 1
            if item.get("type") == "response_item" and kind in ("custom_tool_call", "function_call"):
                tools += 1
    return calls, tools


def _antigravity_steps(session: Path) -> tuple[int, int] | None:
    # one `PLANNER_RESPONSE` step per model response, carrying its tool calls
    transcripts = sorted(session.glob("*/.system_generated/logs/transcript_full.jsonl"))
    if not transcripts:
        return None
    calls = tools = 0
    for transcript in transcripts:
        for item in _lines(transcript):
            if item.get("type") != "PLANNER_RESPONSE":
                continue
            calls += 1
            tools += len(item.get("tool_calls") or [])
    return calls, tools


def _stirrup_steps(session: Path) -> tuple[int, int] | None:
    # one `step` line per model response; `tool_calls` is cumulative within a session file
    transcripts = sorted(session.rglob("[0-9]*.jsonl"))
    if not transcripts:
        return None
    calls = tools = 0
    for transcript in transcripts:
        steps = [item for item in _lines(transcript) if item.get("type") == "step"]
        calls += len(steps)
        tools += steps[-1]["tool_calls"] if steps else 0
    return calls, tools


STEP_READERS = {"codex": _codex_steps, "antigravity": _antigravity_steps, "stirrup": _stirrup_steps}


def summarize_steps(session_root: Path, family: str) -> dict | None:
    """Return `steps` and `tool_calls` of one run's session, or None without a transcript."""

    session = session_root / agent(family).session_dir
    if family == "claude":
        found = _claude_steps(session_root)
    elif family == "stirrup":
        found = _stirrup_steps(session_root)
    elif session.is_dir():
        found = STEP_READERS[family](session)
    else:
        return None
    if found is None:
        return None
    return {"steps": found[0], "tool_calls": found[1]}
