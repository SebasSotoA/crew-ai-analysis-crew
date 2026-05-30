from __future__ import annotations

import json
import re
from typing import Any

from crew_ai_analysis_crew.schemas import NerResult, SentimentResult, SimilarityResult


def extract_json_blob(text: str) -> dict[str, Any]:
    """Parse JSON from raw task output, including fenced code blocks."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty task output")

    if stripped.startswith("{"):
        return json.loads(stripped)

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1).strip())

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return json.loads(stripped[start : end + 1])

    raise ValueError("No JSON object found in task output")


def _find_output_by_key(outputs: list[str], key: str) -> str | None:
    for raw in outputs:
        if key in raw:
            return raw
    return None


def _raw_from_task_output(task_output: Any) -> str:
    return getattr(task_output, "raw", str(task_output))


def parse_task_outputs(task_outputs: list[Any]) -> tuple[NerResult, SentimentResult, SimilarityResult]:
    ner_raw: str | None = None
    sent_raw: str | None = None
    sim_raw: str | None = None

    for task_output in task_outputs:
        raw = _raw_from_task_output(task_output).strip()
        if not raw:
            continue
        try:
            data = extract_json_blob(raw)
        except ValueError:
            continue

        if "entities_by_label" in data and ner_raw is None:
            ner_raw = raw
        elif "scores" in data and "label" in data and sent_raw is None:
            sent_raw = raw
        elif "matches" in data and sim_raw is None:
            sim_raw = raw

    if not ner_raw or not sent_raw or not sim_raw:
        raise ValueError(
            "Could not identify NER, sentiment, and similarity outputs among task results."
        )

    return (
        NerResult.model_validate(extract_json_blob(ner_raw)),
        SentimentResult.model_validate(extract_json_blob(sent_raw)),
        SimilarityResult.model_validate(extract_json_blob(sim_raw)),
    )


def resolve_executive_narrative(crew_result: Any, task_outputs: list[Any]) -> str:
    """Prefer the crew's final output; fall back to the last non-JSON task output."""
    final = (getattr(crew_result, "raw", None) or "").strip()
    if final and not final.startswith("{") and "<html" not in final[:200].lower():
        return final

    for task_output in reversed(task_outputs):
        name = (getattr(task_output, "name", None) or "").lower()
        if "executive" in name or "summary" in name:
            raw = _raw_from_task_output(task_output).strip()
            if raw:
                return raw

    for task_output in reversed(task_outputs):
        raw = _raw_from_task_output(task_output).strip()
        if not raw or raw.startswith("{"):
            continue
        try:
            extract_json_blob(raw)
            continue
        except ValueError:
            return raw

    return final or "_Executive summary was not captured. Re-run the crew or check task logs._"
