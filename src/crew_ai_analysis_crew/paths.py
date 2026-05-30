"""Project-root paths for documents, corpus, and output."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CORPUS_DIR = PROJECT_ROOT / "corpus"
OUTPUT_DIR = PROJECT_ROOT / "output"
HTML_REPORT_PATH = OUTPUT_DIR / "analysis_report.html"
DEFAULT_DOCUMENT_PATH = DOCUMENTS_DIR / "example_article.txt"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if not absolute."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()
