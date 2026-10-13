"""Pydantic 数据模型：定义文献元数据、检索范围、规划、分析与综述答案结构。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """基础模型：禁止未声明字段，防止 LLM 结构化输出引入多余数据。"""

    model_config = ConfigDict(extra="forbid")


class PaperMetadata(StrictModel):
    """一篇文献的基础元数据。"""

    title: str = Field(min_length=1)
    abstract: str = ""
    authors: str = "Unknown"
    year: int | None = Field(default=None, ge=1800, le=2200)
    source: str
    paper_id: str


class ResearchScope(StrictModel):
    """检索范围：可按来源（文献名/作者/路径）与年份区间过滤。"""

    source_filter: str | None = None
    from_year: int | None = Field(default=None, ge=1800, le=2200)
    to_year: int | None = Field(default=None, ge=1800, le=2200)


class ResearchPlan(StrictModel):
    """意图解析结果：是否需要检索、检索式、澄清问题与检索范围。"""

    intent_summary: str
    requires_retrieval: bool = True
    queries: list[str] = Field(default_factory=list, max_length=4)
    clarification_question: str | None = None
    scope: ResearchScope = Field(default_factory=ResearchScope)


class SearchInput(StrictModel):
    """search_literature 检索工具的入参。"""

    query: str = Field(min_length=2)
    source_filter: str | None = None
    from_year: int | None = Field(default=None, ge=1800, le=2200)
    to_year: int | None = Field(default=None, ge=1800, le=2200)


class ResearchAnalysis(StrictModel):
    """证据分析结果：是否充分、摘要、冲突线索与后续检索式。"""

    sufficient: bool
    evidence_summary: str
    conflict_notes: list[str] = Field(default_factory=list, max_length=8)
    followup_queries: list[str] = Field(default_factory=list, max_length=3)


class EvidenceClaim(StrictModel):
    """一条带引用的主张，区分文献观点/综合推断/观点冲突/局限说明。"""

    category: Literal["文献观点", "综合推断", "观点冲突", "局限说明"]
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class AnswerDraft(StrictModel):
    """综述初稿：正文主张、冲突与局限说明分别存放。"""

    claims: list[EvidenceClaim] = Field(min_length=1, max_length=12)
    conflicts: list[EvidenceClaim] = Field(default_factory=list, max_length=8)
    limitations: list[EvidenceClaim] = Field(default_factory=list, max_length=8)


class Citation(StrictModel):
    """渲染进最终答案的一条引用来源。"""

    citation_id: str
    title: str
    authors: str
    year: int | None = None
    source: str
    excerpt: str
