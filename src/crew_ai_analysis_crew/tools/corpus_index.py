from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class CorpusDocument:
    path: str
    text: str
    embedding: np.ndarray


class CorpusIndex:
    """Singleton store for corpus embeddings built at kickoff."""

    _instance: CorpusIndex | None = None

    def __init__(self) -> None:
        self.model: SentenceTransformer | None = None
        self.documents: list[CorpusDocument] = []
        self.model_name: str = ""

    @classmethod
    def get(cls) -> CorpusIndex:
        if cls._instance is None:
            cls._instance = CorpusIndex()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def set(
        self,
        model: SentenceTransformer,
        documents: list[CorpusDocument],
        model_name: str,
    ) -> None:
        self.model = model
        self.documents = documents
        self.model_name = model_name

    @property
    def is_ready(self) -> bool:
        return self.model is not None and len(self.documents) > 0

    @property
    def size(self) -> int:
        return len(self.documents)
