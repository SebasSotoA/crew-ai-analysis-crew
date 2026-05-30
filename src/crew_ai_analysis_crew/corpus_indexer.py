from __future__ import annotations

import logging
import os
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from crew_ai_analysis_crew.paths import resolve_project_path
from crew_ai_analysis_crew.tools.corpus_index import CorpusDocument, CorpusIndex

logger = logging.getLogger(__name__)

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def get_sentence_transformer_model_name() -> str:
    return os.environ.get("SENTENCE_TRANSFORMER_MODEL", DEFAULT_ST_MODEL)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8").strip()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _resolve_corpus_dir(corpus_dir: str | Path) -> Path:
    return resolve_project_path(corpus_dir)


def _collect_corpus_files(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.exists():
        return []
    files: list[Path] = []
    for path in corpus_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def build_corpus_index(
    corpus_dir: str | Path = "corpus",
    model_name: str | None = None,
) -> CorpusIndex:
    """Load corpus files, embed them, and populate the CorpusIndex singleton."""
    model_name = model_name or get_sentence_transformer_model_name()
    corpus_path = _resolve_corpus_dir(corpus_dir)
    index = CorpusIndex.get()

    model = SentenceTransformer(model_name)
    documents: list[CorpusDocument] = []

    for file_path in _collect_corpus_files(corpus_path):
        try:
            text = load_document(file_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", file_path, exc)
            continue
        if not text:
            logger.warning("Skipping empty file: %s", file_path)
            continue
        embedding = model.encode(text, convert_to_numpy=True)
        documents.append(
            CorpusDocument(
                path=str(file_path.relative_to(corpus_path)),
                text=text,
                embedding=embedding,
            )
        )

    index.set(model=model, documents=documents, model_name=model_name)
    logger.info(
        "Indexed %d document(s) from %s using %s",
        len(documents),
        corpus_path,
        model_name,
    )
    return index


def load_uploaded_document(document_path: str | Path) -> str:
    path = resolve_project_path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    text = load_document(path)
    if not text:
        raise ValueError(f"Document is empty or unreadable: {path}")
    return text
