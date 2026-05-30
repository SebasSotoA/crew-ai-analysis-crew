import os
from pathlib import Path
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, after_kickoff, before_kickoff, crew, task
from crewai.tasks.task_output import TaskOutput

from crew_ai_analysis_crew.analysis_store import AnalysisStore
from crew_ai_analysis_crew.corpus_indexer import (
    build_corpus_index,
    load_uploaded_document,
)
from crew_ai_analysis_crew.paths import HTML_REPORT_PATH, OUTPUT_DIR
from crew_ai_analysis_crew.report import (
    build_html_report,
    parse_task_outputs,
    resolve_executive_narrative,
)
from crew_ai_analysis_crew.tools import NerTool, SentimentTool, SimilarityTool
from crew_ai_analysis_crew.tools.corpus_index import CorpusIndex


def _capture_ner_output(output: TaskOutput) -> str:
    AnalysisStore.get().ner_raw = output.raw
    return output.raw


def _capture_sentiment_output(output: TaskOutput) -> str:
    AnalysisStore.get().sentiment_raw = output.raw
    return output.raw


def _capture_similarity_output(output: TaskOutput) -> str:
    AnalysisStore.get().similarity_raw = output.raw
    return output.raw


@CrewBase
class DocumentIntelligenceCrew:
    """Multi-task document intelligence crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    _document_title: str = "document"

    @before_kickoff
    def prepare_inputs(self, inputs: dict) -> dict:
        document_path = inputs.get("document_path") or os.environ.get("DOCUMENT_PATH")
        if not document_path:
            raise ValueError(
                "DOCUMENT_PATH is required. Set the environment variable or pass document_path in inputs."
            )

        corpus_dir = inputs.get("corpus_dir") or os.environ.get("CORPUS_DIR", "corpus")

        AnalysisStore.reset()
        CorpusIndex.reset()
        inputs["document_path"] = document_path
        inputs["corpus_dir"] = corpus_dir
        inputs["document_text"] = load_uploaded_document(document_path)
        inputs["document_title"] = Path(document_path).name
        self._document_title = inputs["document_title"]
        build_corpus_index(corpus_dir=corpus_dir)

        return inputs

    @after_kickoff
    def publish_html_report(self, result):
        """Build HTML from real tool JSON + LLM executive narrative."""
        task_outputs = list(result.tasks_output)
        ner, sentiment, similarity = parse_task_outputs(task_outputs)
        narrative = resolve_executive_narrative(result, task_outputs)

        html = build_html_report(
            document_title=self._document_title,
            ner=ner,
            sentiment=sentiment,
            similarity=similarity,
            executive_narrative=narrative,
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        HTML_REPORT_PATH.write_text(html, encoding="utf-8")
        print(f"\nHTML report written to: {HTML_REPORT_PATH.resolve()}")
        return result

    @agent
    def ner_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["ner_specialist"],  # type: ignore[index]
            tools=[NerTool()],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["sentiment_analyst"],  # type: ignore[index]
            tools=[SentimentTool()],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def similarity_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["similarity_analyst"],  # type: ignore[index]
            tools=[SimilarityTool()],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def executive_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["executive_summarizer"],  # type: ignore[index]
            verbose=True,
            allow_delegation=False,
        )

    @task
    def ner_task(self) -> Task:
        return Task(
            config=self.tasks_config["ner_task"],  # type: ignore[index]
            callback=_capture_ner_output,
        )

    @task
    def sentiment_task(self) -> Task:
        return Task(
            config=self.tasks_config["sentiment_task"],  # type: ignore[index]
            callback=_capture_sentiment_output,
        )

    @task
    def similarity_task(self) -> Task:
        return Task(
            config=self.tasks_config["similarity_task"],  # type: ignore[index]
            callback=_capture_similarity_output,
        )

    @task
    def executive_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["executive_summary_task"],  # type: ignore[index]
            context=[
                self.ner_task(),
                self.sentiment_task(),
                self.similarity_task(),
            ],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
