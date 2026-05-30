#!/usr/bin/env python
import os
import sys
import warnings

from crew_ai_analysis_crew.crew import DocumentIntelligenceCrew
from crew_ai_analysis_crew.paths import (
    CORPUS_DIR as DEFAULT_CORPUS_DIR,
    DEFAULT_DOCUMENT_PATH,
    resolve_project_path,
)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def _base_inputs() -> dict:
    doc_setting = os.environ.get("DOCUMENT_PATH", "").strip()
    document_path = resolve_project_path(
        doc_setting if doc_setting else DEFAULT_DOCUMENT_PATH
    )
    if not document_path.exists():
        raise FileNotFoundError(
            f"Document not found: {document_path}. "
            "Add a .txt or .pdf under documents/ and set DOCUMENT_PATH in .env"
        )

    corpus_setting = os.environ.get("CORPUS_DIR", "").strip()
    corpus_dir = resolve_project_path(
        corpus_setting if corpus_setting else DEFAULT_CORPUS_DIR
    )

    return {
        "document_path": str(document_path),
        "corpus_dir": str(corpus_dir),
    }


def run():
    """Run the document intelligence crew."""
    inputs = _base_inputs()
    try:
        DocumentIntelligenceCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}") from e


def train():
    """Train the crew for a given number of iterations."""
    inputs = _base_inputs()
    try:
        DocumentIntelligenceCrew().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}") from e


def replay():
    """Replay the crew execution from a specific task."""
    try:
        DocumentIntelligenceCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


def test():
    """Test the crew execution and returns the results."""
    inputs = _base_inputs()
    try:
        DocumentIntelligenceCrew().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}") from e


def run_with_trigger():
    """Run the crew with trigger payload."""
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = _base_inputs()
    inputs["crewai_trigger_payload"] = trigger_payload

    try:
        return DocumentIntelligenceCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}") from e


if __name__ == "__main__":
    run()
