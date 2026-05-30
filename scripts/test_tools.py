#!/usr/bin/env python
"""Quick smoke test for ML tools without running the full crew."""
import os

from crew_ai_analysis_crew.corpus_indexer import build_corpus_index, load_uploaded_document
from crew_ai_analysis_crew.paths import CORPUS_DIR, DEFAULT_DOCUMENT_PATH
from crew_ai_analysis_crew.tools.corpus_index import CorpusIndex
from crew_ai_analysis_crew.tools.ner_tool import NerTool
from crew_ai_analysis_crew.tools.sentiment_tool import SentimentTool
from crew_ai_analysis_crew.tools.similarity_tool import SimilarityTool


def main() -> None:
    doc_path = os.environ.get("DOCUMENT_PATH", str(DEFAULT_DOCUMENT_PATH))
    corpus_dir = os.environ.get("CORPUS_DIR", str(CORPUS_DIR))

    CorpusIndex.reset()
    text = load_uploaded_document(doc_path)
    build_corpus_index(corpus_dir=corpus_dir)

    ner = NerTool()
    sentiment = SentimentTool()
    similarity = SimilarityTool()

    print("=== NER ===")
    print(ner.run(document_text=text)[:500])
    print("\n=== Sentiment ===")
    print(sentiment.run(document_text=text))
    print("\n=== Similarity ===")
    print(similarity.run(document_text=text))


if __name__ == "__main__":
    main()
