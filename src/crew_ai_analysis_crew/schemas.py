from pydantic import BaseModel, Field


class EntityRecord(BaseModel):
    label: str
    text: str
    start: int
    end: int


class NerResult(BaseModel):
    entities_by_label: dict[str, list[str]]
    entity_count: int
    entities: list[EntityRecord] = Field(default_factory=list)


class SentimentResult(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]
    insights: str


class SimilarDocument(BaseModel):
    path: str
    score: float
    snippet: str


class SimilarityResult(BaseModel):
    matches: list[SimilarDocument]
    corpus_size: int
    message: str = ""


class ExecutiveSummaryReport(BaseModel):
    title: str
    summary: str
    key_findings: list[str]
    risks_or_concerns: list[str] = Field(default_factory=list)
