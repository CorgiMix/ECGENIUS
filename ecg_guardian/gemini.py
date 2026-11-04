"""Gemini: AF verdict + score + clinician summary, either alone or reading Jev's answers.

Gemini is non-deterministic, so: temperature 0, pinned model name, every response cached and logged.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .config import GEMINI_MODEL, LEADS

PROMPT_VERSION = "v4"

SYSTEM_ALONE = (
    "You read summaries of 10-second 12-lead ECGs and decide whether the ECG shows atrial "
    "fibrillation (AF). Features are pre-computed and bucketed: rhythm regularity, heart rate, "
    "signal noise, which leads are bad and, when present, P-wave consistency and the irregularity "
    "pattern. Irregular rhythm in a noisy window is weak evidence; consistent P waves, or irregularity "
    "explained by early beats with pauses, argue against AF; 'irregularly irregular' argues for it. "
    "Give the probability of AF and a 2-3 sentence summary for a clinician. Only mention windows, "
    "leads and findings that appear in the input."
)
SYSTEM_WITH_JEV = SYSTEM_ALONE + (
    " You are also given the answers of a decision model (Jev) about the same ECG; Jev's numbers "
    "are probabilities between 0 and 1. Use them as evidence, and say where you disagree with Jev."
)


class Verdict(BaseModel):
    af_percent: int = Field(ge=0, le=100, description="Probability (0-100) that this ECG shows AF")
    action: str = Field(description="'trust' if the call is clear, 'review' if a clinician should look")
    summary: str = Field(description="2-3 sentences for a clinician, naming the windows/leads that matter")


def build_prompt(state: dict, jev_answers: dict | None) -> str:
    payload = {"ecg_summary": state}
    if jev_answers is not None:
        payload["jev_answers"] = {k: v for k, v in jev_answers.items() if k not in ("model", "input_tokens")}
    return json.dumps(payload, indent=1)


def _key(system: str, prompt: str) -> str:
    return hashlib.sha256(f"{GEMINI_MODEL}|{PROMPT_VERSION}|{system}|{prompt}".encode()).hexdigest()[:16]


async def ask_many(items: list[tuple[int, dict, dict | None]], cache_file: Path, concurrency: int = 8,
                   log=print, progress=None) -> dict[int, dict]:
    """items = [(ecg_id, state, jev_answers or None)]. Returns {ecg_id: verdict}, cached in jsonl."""
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); cache[row["key"]] = row["verdict"]
    results, todo = {}, []
    for ecg_id, state, ans in items:
        system = SYSTEM_ALONE if ans is None else SYSTEM_WITH_JEV
        prompt = build_prompt(state, ans); k = _key(system, prompt)
        if k in cache:
            results[ecg_id] = cache[k]
        else:
            todo.append((ecg_id, system, prompt, k))
    log(f"gemini: {len(results)} cached, {len(todo)} to request")
    if not todo:
        return results
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sem = asyncio.Semaphore(concurrency)

    async def one(ecg_id, system, prompt, k):
        config = types.GenerateContentConfig(system_instruction=system, temperature=0.0,
                                             response_mime_type="application/json", response_schema=Verdict)
        async with sem:
            for attempt in range(6):
                try:
                    r = await client.aio.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
                    v = r.parsed if isinstance(r.parsed, Verdict) else Verdict.model_validate_json(r.text)
                    u = r.usage_metadata
                    return ecg_id, k, {**v.model_dump(), "model": GEMINI_MODEL,
                                       "input_tokens": getattr(u, "prompt_token_count", None),
                                       "output_tokens": getattr(u, "candidates_token_count", None)}
                except Exception as e:  # rate limits / transient errors: back off and retry
                    if attempt == 5:
                        log(f"gemini failed for {ecg_id}: {type(e).__name__}: {str(e)[:120]}")
                        return ecg_id, k, None
                    await asyncio.sleep(2 ** attempt)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("a", encoding="utf-8") as f:
        for coro in asyncio.as_completed([one(*t) for t in todo]):
            ecg_id, k, v = await coro
            if v is not None:
                results[ecg_id] = v
                f.write(json.dumps({"key": k, "ecg_id": int(ecg_id), "verdict": v}) + "\n")
            if progress:
                progress(1)
    return results


def unsupported_leads(summary: str, state: dict) -> list[str]:
    """Faithfulness check: leads Gemini names as problems that no window lists as bad."""
    windows = state.get("segments") or [state["ecg"]]
    flagged = {l.strip() for w in windows if w["bad_leads"] != "none" for l in w["bad_leads"].split(",")}
    tokens = {t.strip(".,;:()'\"") for t in summary.replace("/", " ").split()}
    return sorted(l for l in LEADS if len(l) > 1 and l in tokens and l not in flagged)
