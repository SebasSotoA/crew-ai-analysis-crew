# Multi-Task Document Intelligence Crew

A CrewAI pipeline that analyzes uploaded documents using specialized agents:

- **NER** — spaCy `en_core_web_sm`
- **Sentiment** — `cardiffnlp/twitter-roberta-base-sentiment`
- **Similarity** — `sentence-transformers/all-MiniLM-L6-v2` against a dynamic `corpus/`
- **Executive HTML report** — LLM writes a detailed markdown analysis; the pipeline builds HTML with **real** charts, **all entities**, and full corpus snippets from tool JSON

## Prerequisites

- Python 3.10–3.13
- [UV](https://docs.astral.sh/uv/) package manager
- `OPENAI_API_KEY` in `.env` (for the executive summary agent)

## Installation

```powershell
cd crew-ai-analysis-crew
uv sync --prerelease=allow
```

The spaCy English model (`en_core_web_sm`) is installed automatically as a project dependency.

First run downloads Hugging Face model weights (sentiment + sentence-transformers). This can take several minutes.

## Project layout

Everything lives under this repo:

```
crew-ai-analysis-crew/
├── documents/     # Main file to analyze (news article, report, etc.)
├── corpus/        # Reference library for similarity search
├── output/        # Generated HTML report (gitignored)
├── .env           # API keys + DOCUMENT_PATH + CORPUS_DIR
└── src/crew_ai_analysis_crew/
```

| Folder | Purpose |
|--------|---------|
| [`documents/`](documents/) | **One primary document per run** — e.g. today's election news |
| [`corpus/`](corpus/) | **Background archive** — older articles, profiles, explainers (re-indexed each run) |
| [`output/`](output/) | Agent-generated HTML report (`analysis_report.html`) |

## Usage

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Put your **main article** in `documents/` (e.g. `documents/election_news.txt`).
3. Put **reference articles** in `corpus/`.
4. Update `.env`:

```env
DOCUMENT_PATH=documents/election_news.txt
CORPUS_DIR=corpus
```

5. Run:

```powershell
crewai run
```

6. Open [`output/analysis_report.html`](output/analysis_report.html) in your browser.

Defaults (if omitted in `.env`): `documents/example_article.txt` and `corpus/`.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | LLM API key for executive summary |
| `DOCUMENT_PATH` | No | `documents/example_article.txt` | Main document to analyze |
| `CORPUS_DIR` | No | `corpus` | Reference documents folder |
| `SENTENCE_TRANSFORMER_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model override |
| `MODEL` | No | — | OpenAI model for agents (e.g. `gpt-4o-mini`) |

Paths may be relative to the project root or absolute.

## Notes

- PDF support uses text extraction only (no OCR for scanned documents).
- NER, sentiment, and similarity tasks run in parallel; first run loads multiple ML models and may use significant RAM.
- If memory is limited, set `async_execution: false` on the three specialist tasks in `config/tasks.yaml`.

## Support

- [CrewAI documentation](https://docs.crewai.com)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
