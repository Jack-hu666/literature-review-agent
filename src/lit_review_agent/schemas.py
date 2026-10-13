from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PaperMetadata(StrictModel):
    title: str = Field(min_length=1)
    abstract: str = ""
    authors: str = "Unknown"
    year: int | None = Field(default=None, ge=1800, le=2200)
    source: str
    paper_id: str


class ResearchScope(StrictModel):
    source_filter: str | None = None
    from_year: int | None = Field(default=None, ge=1800, le=2200)
    to_year: int | None = Field(default=None, ge=1800, le=2200)


class ResearchPlan(StrictModel):
    intent_summary: str
    requires_retrieval: bool = True
    queries: list[str] = Field(default_factory=list, max_length=4)
    clarification_question: str | None = None
    scope: ResearchScope = Field(default_factory=ResearchScope)


class SearchInput(StrictModel):
    query: str = Field(min_length=2)
    source_filter: str | None = None
    from_year: int | None = Field(default=None, ge=1800, le=2200)
    to_year: int | None = Field(default=None, ge=1800, le=2200)


class ResearchAnalysis(StrictModel):
    sufficient: bool
    evidence_summary: str
    conflict_notes: list[str] = Field(default_factory=list, max_length=8)
    followup_queries: list[str] = Field(default_factory=list, max_length=3)


class EvidenceClaim(StrictModel):
    category: Literal["文献观点", "综合推断", "观点冲突", "局限说明"]
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class AnswerDraft(StrictModel):
    claims: list[EvidenceClaim] = Field(min_length=1, max_length=12)
    conflicts: list[EvidenceClaim] = Field(default_factory=list, max_length=8)
    limitations: list[EvidenceClaim] = Field(default_factory=list, max_length=8)


class Citation(StrictModel):
    citation_id: str
    title: str
    authors: str
    year: int | None = None
    source: str
    excerpt: str
