import json
from collections import defaultdict
from typing import Any, Type

import spacy
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew_ai_analysis_crew.schemas import EntityRecord, NerResult

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


class NerToolInput(BaseModel):
    document_text: str = Field(..., description="Full text of the document to analyze.")


class NerTool(BaseTool):
    name: str = "extract_named_entities"
    description: str = (
        "Extract named entities (people, organizations, locations, dates, etc.) "
        "from document text using spaCy NER."
    )
    args_schema: Type[BaseModel] = NerToolInput

    def _run(self, document_text: str, **kwargs: Any) -> str:
        _ = kwargs
        nlp = _get_nlp()
        doc = nlp(document_text)
        entities_by_label: dict[str, list[str]] = defaultdict(list)
        entities: list[EntityRecord] = []

        for ent in doc.ents:
            entities_by_label[ent.label_].append(ent.text)
            entities.append(
                EntityRecord(
                    label=ent.label_,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )

        for label in entities_by_label:
            entities_by_label[label] = sorted(set(entities_by_label[label]))

        result = NerResult(
            entities_by_label=dict(entities_by_label),
            entity_count=len(entities),
            entities=entities,
        )
        return json.dumps(result.model_dump(), indent=2)
