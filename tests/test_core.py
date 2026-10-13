import unittest

from langchain_core.documents import Document

from lit_review_agent.graph import _make_citation, _render_answer, build_graph
from lit_review_agent.ingestion import extract_metadata
from lit_review_agent.retrieval import _tokenize
from lit_review_agent.schemas import (
    AnswerDraft,
    EvidenceClaim,
    ResearchAnalysis,
    ResearchPlan,
    ResearchScope,
)


class FakeRetriever:
    def __init__(self):
        self.calls = []
        self.document = Document(
            page_content="Dense search finds paraphrases; BM25 is useful for exact technical terms.",
            metadata={
                "chunk_id": "chunk0001abcdef",
                "paper_id": "paper001",
                "title": "Hybrid Retrieval Demo",
                "authors": "Demo Research Group",
                "year": 2022,
                "source": "demo_study_2022.pdf",
            },
        )

    def search(self, query, scope=None):
        self.calls.append((query, scope))
        return [self.document]


class FakeStructuredLLM:
    def __init__(self):
        self.analysis_calls = 0

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, _messages):
        if self.schema is ResearchPlan:
            return ResearchPlan(intent_summary="比较检索方法", queries=["retrieval methods"])
        if self.schema is ResearchAnalysis:
            self.analysis_calls += 1
            if self.analysis_calls == 1:
                return ResearchAnalysis(
                    sufficient=False,
                    evidence_summary="仍需检索评价限制",
                    followup_queries=["retrieval evaluation limitations"],
                )
            return ResearchAnalysis(sufficient=True, evidence_summary="找到可用于演示的证据")
        if self.schema is AnswerDraft:
            return AnswerDraft(
                claims=[EvidenceClaim(
                    category="文献观点",
                    text="该样例认为语义检索有助于处理改写问题。",
                    citation_ids=["Cchunk0001abcdef"],
                )],
            )
        raise AssertionError(f"Unexpected output schema: {self.schema}")


class FakeBadCitationLLM(FakeStructuredLLM):
    def invoke(self, messages):
        if self.schema is AnswerDraft:
            return AnswerDraft(
                claims=[EvidenceClaim(category="综合推断", text="无证据", citation_ids=["NOT_REAL"])],
            )
        return super().invoke(messages)


class CoreTests(unittest.TestCase):
    def test_extracts_explicit_paper_metadata(self):
        paper = extract_metadata(
            "# Title: A Retrieval Study\nAuthors: Lee, Kim\nYear: 2023\nAbstract: Evidence summary here.\n\n## Methods\nBody.",
            "sample.md",
        )
        self.assertEqual(paper.title, "A Retrieval Study")
        self.assertEqual(paper.authors, "Lee, Kim")
        self.assertEqual(paper.year, 2023)
        self.assertIn("Evidence summary", paper.abstract)

    def test_metadata_joins_wrapped_pdf_title(self):
        paper = extract_metadata(
            "Title: Hybrid Retrieval for Small Research\nCollections: A Demonstration Study\n"
            "Authors: Demo Research Group\nYear: 2022\nAbstract: Sample abstract.",
            "sample.pdf",
        )
        self.assertEqual(paper.title, "Hybrid Retrieval for Small Research Collections: A Demonstration Study")

    def test_chinese_keyword_tokenizer_emits_overlapping_bigrams(self):
        self.assertEqual(_tokenize("混合检索 BM25"), ["混合", "合检", "检索", "bm25"])

    def test_graph_rewrites_query_and_respects_user_scope(self):
        llm = FakeStructuredLLM()
        retriever = FakeRetriever()
        graph = build_graph(llm, retriever, max_iterations=2)
        result = graph.invoke({
            "question": "Compare retrieval approaches",
            "scope": ResearchScope(source_filter="Demo Research Group", from_year=2020, to_year=2024),
        }, config={"configurable": {"thread_id": "test-session"}})

        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(retriever.calls[1][0], "retrieval evaluation limitations")
        self.assertEqual(retriever.calls[0][1].source_filter, "Demo Research Group")
        self.assertEqual(retriever.calls[0][1].from_year, 2020)
        self.assertIn("[Cchunk0001abcdef]", result["final_answer"])
        self.assertEqual(result["iteration"], 2)

    def test_citation_rendering_omits_unknown_ids(self):
        document = FakeRetriever().document
        citation = _make_citation({"page_content": document.page_content, "metadata": document.metadata})
        draft = AnswerDraft(
            claims=[EvidenceClaim(category="文献观点", text="Supported.", citation_ids=[citation.citation_id, "INVALID"])],
        )
        rendered = _render_answer(draft, [citation])
        self.assertIn(f"[{citation.citation_id}]", rendered)
        self.assertNotIn("INVALID", rendered)

    def test_invalid_model_citation_falls_back_to_excerpt(self):
        graph = build_graph(FakeBadCitationLLM(), FakeRetriever(), max_iterations=1)
        result = graph.invoke(
            {"question": "Summarize retrieval", "scope": ResearchScope()},
            config={"configurable": {"thread_id": "bad-citation-session"}},
        )
        self.assertIn("未通过引用校验", result["final_answer"])
        self.assertIn("[Cchunk0001abcdef]", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
