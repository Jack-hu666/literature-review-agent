"""混合检索：以倒数排名融合（RRF）合并 Chroma 语义检索与 BM25 关键词检索。"""

import logging
import re
from collections import defaultdict

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma

from lit_review_agent.schemas import ResearchScope

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """中文按重叠二元组、英文按小写词切分，供 BM25 关键词召回使用。"""
    tokens: list[str] = []
    for segment in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.casefold()):
        if segment.isascii():
            tokens.append(segment)
        elif len(segment) == 1:
            tokens.append(segment)
        else:
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


class HybridRetriever:
    """以倒数排名融合（RRF）合并 Chroma 语义检索与 BM25 关键词检索结果。"""

    def __init__(self, vector_store: Chroma, top_k: int = 6, rrf_constant: int = 60) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.rrf_constant = rrf_constant

    def _all_documents(self) -> list[Document]:
        """从 Chroma 取出全部已入库片段及其元数据。"""
        stored = self.vector_store.get(include=["documents", "metadatas"])
        contents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        return [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(contents, metadatas)
            if content
        ]

    @staticmethod
    def _matches_scope(document: Document, scope: ResearchScope) -> bool:
        """按来源/标题/作者与年份范围判断片段是否满足检索范围。"""
        metadata = document.metadata
        if scope.source_filter:
            needle = scope.source_filter.casefold()
            searchable = " ".join(str(metadata.get(key, "")) for key in ("source", "title", "authors"))
            if needle not in searchable.casefold():
                return False
        year = int(metadata.get("year") or 0)
        if scope.from_year is not None and (not year or year < scope.from_year):
            return False
        if scope.to_year is not None and (not year or year > scope.to_year):
            return False
        return True

    def search(self, query: str, scope: ResearchScope | None = None) -> list[Document]:
        """执行混合检索并按 RRF 分数返回前 top_k 个片段。"""
        scope = scope or ResearchScope()
        corpus = self._all_documents()
        if not corpus:
            return []

        candidates: dict[str, Document] = {}
        scores: defaultdict[str, float] = defaultdict(float)

        def add_ranked(documents: list[Document]) -> None:
            for rank, document in enumerate(documents, start=1):
                if not self._matches_scope(document, scope):
                    continue
                key = str(document.metadata.get("chunk_id") or hash(document.page_content))
                candidates[key] = document
                scores[key] += 1.0 / (self.rrf_constant + rank)

        semantic = self.vector_store.similarity_search(query, k=min(len(corpus), self.top_k * 4))
        add_ranked(semantic)

        try:
            keyword = BM25Retriever.from_documents(
                corpus,
                k=min(len(corpus), self.top_k * 4),
                preprocess_func=_tokenize,
            )
            add_ranked(keyword.invoke(query))
        except Exception as exc:
            logger.warning("BM25 检索不可用，仅返回向量检索结果: %s", exc)

        ordered = sorted(candidates, key=lambda key: scores[key], reverse=True)
        return [candidates[key] for key in ordered[: self.top_k]]
