import json
from typing import Any, Type

import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_analysis_crew.schemas import SimilarDocument, SimilarityResult
from crew_ai_analysis_crew.tools.corpus_index import CorpusIndex

TOP_K = 5
SNIPPET_LEN = 300


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SimilarityToolInput(BaseModel):
    document_text: str = Field(..., description="Full text of the document to compare against the corpus.")
    top_k: int = Field(default=TOP_K, description="Number of similar documents to return.")


class SimilarityTool(BaseTool):
    name: str = "find_similar_documents"
    description: str = (
        "Find the most semantically similar documents in the indexed corpus "
        "using sentence-transformers/all-MiniLM-L6-v2 embeddings."
    )
    args_schema: Type[BaseModel] = SimilarityToolInput

    def _run(self, document_text: str, top_k: int = TOP_K, **kwargs: Any) -> str:
        _ = kwargs
        index = CorpusIndex.get()

        if not index.is_ready:
            result = SimilarityResult(
                matches=[],
                corpus_size=0,
                message=(
                    "No documents indexed. Add .txt or .pdf files to the corpus/ "
                    "folder and run again."
                ),
            )
            return json.dumps(result.model_dump(), indent=2)

        query_embedding = index.model.encode(document_text, convert_to_numpy=True)
        scored: list[tuple[float, str, str]] = []

        for doc in index.documents:
            score = _cosine_similarity(query_embedding, doc.embedding)
            snippet = doc.text[:SNIPPET_LEN].replace("\n", " ")
            if len(doc.text) > SNIPPET_LEN:
                snippet += "..."
            scored.append((score, doc.path, snippet))

        scored.sort(key=lambda x: x[0], reverse=True)
        matches = [
            SimilarDocument(path=path, score=round(score, 4), snippet=snippet)
            for score, path, snippet in scored[:top_k]
        ]

        result = SimilarityResult(
            matches=matches,
            corpus_size=index.size,
            message=f"Compared against {index.size} document(s) in corpus.",
        )
        return json.dumps(result.model_dump(), indent=2)
