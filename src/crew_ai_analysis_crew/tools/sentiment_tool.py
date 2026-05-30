import json
from typing import Any, Type, cast

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from crew_ai_analysis_crew.schemas import SentimentResult

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment"
# RoBERTa hard limit is 512 tokens (including special tokens).
MAX_MODEL_TOKENS = 512
CHUNK_TOKEN_SIZE = 500

_sentiment_pipeline = None
_sentiment_tokenizer = None

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def _get_sentiment_pipeline() -> Any:
    global _sentiment_pipeline, _sentiment_tokenizer
    if _sentiment_pipeline is None:
        _sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        # cast: transformers pipeline() overloads are incomplete for sentiment-analysis
        _sentiment_pipeline = cast(Any, pipeline)(
            "sentiment-analysis",
            model=model,
            tokenizer=_sentiment_tokenizer,
            top_k=None,
            truncation=True,
            max_length=MAX_MODEL_TOKENS,
        )
    return _sentiment_pipeline


def _top_sentiment_label(scores: dict[str, float]) -> tuple[str, float]:
    label, confidence = max(scores.items(), key=lambda item: item[1])
    return label, confidence


def _chunk_text_for_model(text: str) -> list[str]:
    """Split long documents into token-safe chunks for RoBERTa."""
    tokenizer = _sentiment_tokenizer or _get_sentiment_pipeline().tokenizer
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= CHUNK_TOKEN_SIZE:
        return [text]

    chunks: list[str] = []
    for start in range(0, len(token_ids), CHUNK_TOKEN_SIZE):
        piece_ids = token_ids[start : start + CHUNK_TOKEN_SIZE]
        chunks.append(tokenizer.decode(piece_ids, skip_special_tokens=True))
    return chunks


def _normalize_label(raw_label: Any) -> str:
    label_key = str(raw_label)
    if label_key in LABEL_MAP:
        return LABEL_MAP[label_key]
    return label_key.lower()


def _scores_from_pipeline_output(raw_scores: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in raw_scores:
        label = _normalize_label(item.get("label", ""))
        scores[label] = round(float(item["score"]), 4)
    return scores


def _aggregate_chunk_scores(chunk_score_lists: list[dict[str, float]]) -> dict[str, float]:
    if len(chunk_score_lists) == 1:
        return chunk_score_lists[0]

    totals = {key: 0.0 for key in ("positive", "negative", "neutral")}
    for scores in chunk_score_lists:
        for key, value in scores.items():
            totals[key] = totals.get(key, 0.0) + value

    count = len(chunk_score_lists)
    return {key: round(value / count, 4) for key, value in totals.items()}


class SentimentToolInput(BaseModel):
    document_text: str = Field(..., description="Full text of the document to analyze.")


class SentimentTool(BaseTool):
    name: str = "analyze_sentiment"
    description: str = (
        "Classify overall document sentiment as positive, negative, or neutral "
        "using the cardiffnlp/twitter-roberta-base-sentiment model."
    )
    args_schema: Type[BaseModel] = SentimentToolInput

    def _run(self, document_text: str, **kwargs: Any) -> str:
        _ = kwargs
        classifier = _get_sentiment_pipeline()
        chunks = _chunk_text_for_model(document_text)

        chunk_scores: list[dict[str, float]] = []
        for chunk in chunks:
            raw_scores = classifier(chunk)[0]
            chunk_scores.append(_scores_from_pipeline_output(raw_scores))

        scores = _aggregate_chunk_scores(chunk_scores)
        top_label, confidence = _top_sentiment_label(scores)

        segment_note = (
            f" Analyzed {len(chunks)} segment(s) (long document split for the 512-token model limit)."
            if len(chunks) > 1
            else ""
        )
        insights = (
            f"Overall sentiment is {top_label} with {confidence:.1%} confidence. "
            f"Score distribution: {', '.join(f'{k}={v:.2%}' for k, v in scores.items())}."
            f"{segment_note}"
        )

        result = SentimentResult(
            label=top_label,
            confidence=confidence,
            scores=scores,
            insights=insights,
        )
        return json.dumps(result.model_dump(), indent=2)
