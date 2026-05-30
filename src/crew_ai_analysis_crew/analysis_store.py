"""Holds raw specialist task outputs for report generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnalysisStore:
    ner_raw: str = ""
    sentiment_raw: str = ""
    similarity_raw: str = ""

    _instance: AnalysisStore | None = None

    @classmethod
    def get(cls) -> AnalysisStore:
        if cls._instance is None:
            cls._instance = AnalysisStore()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = AnalysisStore()

    @property
    def is_complete(self) -> bool:
        return bool(self.ner_raw and self.sentiment_raw and self.similarity_raw)
