"""LangGraph 工作流：意图解析 → 检索 → 证据分析 → 生成带引用的综述。"""

import json
import logging
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from lit_review_agent.retrieval import HybridRetriever
from lit_review_agent.schemas import (
    AnswerDraft,
    Citation,
    ResearchAnalysis,
    ResearchPlan,
    ResearchScope,
    SearchInput,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """贯穿工作流的状态：问题、规划、检索结果、证据分析与最终答案。"""
    question: str
    conversation_history: list[dict[str, str]]
    plan: ResearchPlan
    queries: list[str]
    search_query: str
    scope: ResearchScope
    retrieved_chunks: list[dict[str, Any]]
    analysis: ResearchAnalysis
    reasoning_logs: list[str]
    iteration: int
    max_iterations: int
    final_answer: str
    citations: list[dict[str, Any]]
    route: str


def _as_model(schema: type, response: Any) -> Any:
    """把 LLM 输出规整为指定 Pydantic 模型实例。"""
    return response if isinstance(response, schema) else schema.model_validate(response)


def _doc_to_dict(document: Document) -> dict[str, Any]:
    """把 LangChain Document 转为可 JSON 序列化的字典。"""
    return {"page_content": document.page_content, "metadata": dict(document.metadata)}


def _make_retrieval_tool(retriever: HybridRetriever) -> StructuredTool:
    """把混合检索器包装为 LangChain 结构化工具 search_literature。"""
    def search_literature(
        query: str,
        source_filter: str | None = None,
        from_year: int | None = None,
        to_year: int | None = None,
    ) -> str:
        scope = ResearchScope(source_filter=source_filter, from_year=from_year, to_year=to_year)
        results = retriever.search(query, scope)
        return json.dumps([_doc_to_dict(item) for item in results], ensure_ascii=False)

    return StructuredTool.from_function(
        func=search_literature,
        name="search_literature",
        description="在已导入的文献知识库中进行关键词与语义混合检索，返回带来源元数据的原文片段。",
        args_schema=SearchInput,
    )


def _make_citation(chunk: dict[str, Any]) -> Citation:
    """把检索片段及其元数据转为可渲染的引用对象。"""
    metadata = chunk["metadata"]
    chunk_id = str(metadata.get("chunk_id", "unknown"))
    year = int(metadata.get("year") or 0) or None
    return Citation(
        citation_id=f"C{chunk_id}",
        title=str(metadata.get("title") or "Untitled"),
        authors=str(metadata.get("authors") or "Unknown"),
        year=year,
        source=str(metadata.get("source") or "unknown"),
        excerpt=chunk["page_content"][:700],
    )


def _render_answer(draft: AnswerDraft, citations: list[Citation]) -> str:
    """把综述初稿与引用渲染为带引用标记与来源清单的 Markdown 文本。"""
    by_id = {item.citation_id: item for item in citations}
    lines = ["**文献观点与综合分析**"]

    def render_claim(claim: Any) -> None:
        valid_ids = [citation_id for citation_id in claim.citation_ids if citation_id in by_id]
        if not valid_ids:
            return
        refs = " ".join(f"[{citation_id}]" for citation_id in valid_ids)
        lines.append(f"- [{claim.category}] {claim.text.strip()} {refs}")

    for claim in draft.claims:
        render_claim(claim)
    if draft.conflicts:
        lines.append("\n**观点差异**")
        for claim in draft.conflicts:
            render_claim(claim)
    if draft.limitations:
        lines.append("\n**局限与注意事项**")
        for claim in draft.limitations:
            render_claim(claim)
    all_claims = draft.claims + draft.conflicts + draft.limitations
    used_ids = {citation_id for claim in all_claims for citation_id in claim.citation_ids if citation_id in by_id}
    if used_ids:
        lines.append("\n**引用来源**")
        for citation_id in sorted(used_ids):
            citation = by_id[citation_id]
            year = f", {citation.year}" if citation.year else ""
            lines.append(f"- [{citation_id}] {citation.authors}{year}. {citation.title}. ({citation.source})")
    return "\n".join(lines).strip()


def build_graph(llm: Any, retriever: HybridRetriever, max_iterations: int = 3) -> Any:
    """Build a bounded plan-act-observe-reason LangGraph with in-memory thread memory."""
    search_tool = _make_retrieval_tool(retriever)

    def parse_intent(state: AgentState) -> dict[str, Any]:
        """解析问题意图，生成检索式与检索范围；不需要检索时直接转向回答。"""
        history = state.get("conversation_history", [])[-6:]
        prompt = [
            SystemMessage(content=(
                "你是科研文献综述助手的任务规划器。判断问题是否需要检索文献；研究性结论必须检索。"
                "若问题缺少研究对象或关键范围，提出一个简洁澄清问题。识别用户指定的文献名/作者范围和年份范围，"
                "生成 1-3 个简洁检索式。只输出符合给定结构的数据。"
            )),
            HumanMessage(content=json.dumps({"question": state["question"], "conversation_history": history}, ensure_ascii=False)),
        ]
        plan = _as_model(ResearchPlan, llm.with_structured_output(ResearchPlan).invoke(prompt))
        queries = [query.strip() for query in plan.queries if query.strip()]
        if plan.requires_retrieval and not plan.clarification_question and not queries:
            queries = [state["question"]]
        requested_scope = state.get("scope", ResearchScope())
        scope = ResearchScope(
            source_filter=requested_scope.source_filter or plan.scope.source_filter,
            from_year=requested_scope.from_year if requested_scope.from_year is not None else plan.scope.from_year,
            to_year=requested_scope.to_year if requested_scope.to_year is not None else plan.scope.to_year,
        )
        logs = [f"计划：{plan.intent_summary}", f"初始检索式：{queries or '无需检索'}"]
        logger.info("intent parsed; retrieval=%s, queries=%s", plan.requires_retrieval, queries)
        return {
            "plan": plan,
            "scope": scope,
            "queries": queries,
            "search_query": queries[0] if queries else "",
            "retrieved_chunks": [],
            "reasoning_logs": logs,
            "iteration": 0,
            "max_iterations": max_iterations,
            "citations": [],
            "route": "answer" if plan.clarification_question or not plan.requires_retrieval else "retrieve",
        }

    def decide_tool(state: AgentState) -> dict[str, Any]:
        """把当前 route 透传给条件边。"""
        return {"route": state.get("route", "answer")}

    def retrieve(state: AgentState) -> dict[str, Any]:
        """按当前检索式与范围执行检索，并去重合并到已检索片段。"""
        scope = state.get("scope", ResearchScope())
        result = search_tool.invoke({
            "query": state["search_query"],
            "source_filter": scope.source_filter,
            "from_year": scope.from_year,
            "to_year": scope.to_year,
        })
        found = json.loads(result)
        existing = {item["metadata"].get("chunk_id") for item in state.get("retrieved_chunks", [])}
        combined = list(state.get("retrieved_chunks", []))
        combined.extend(item for item in found if item["metadata"].get("chunk_id") not in existing)
        iteration = state.get("iteration", 0) + 1
        logs = state.get("reasoning_logs", []) + [
            f"Action: search_literature(query={state['search_query']!r})",
            f"Observation: 检索到 {len(found)} 个片段（累计去重 {len(combined)} 个，轮次 {iteration}）。",
        ]
        logger.info("retrieval iteration %d returned %d chunks", iteration, len(found))
        return {"retrieved_chunks": combined, "iteration": iteration, "reasoning_logs": logs}

    def analyze_evidence(state: AgentState) -> dict[str, Any]:
        """分析已检索证据，证据不足且未达轮数上限时改写检索式继续。"""
        prompt = [
            SystemMessage(content=(
                "你是文献证据分析器。只依据提供的文献片段分析，不得补充常识或编造结论。"
                "比较多篇文献方法与发现；若结论冲突，明确描述冲突及可能的研究条件差异，不强行裁定。"
                "若证据不足且仍可通过新关键词检索，给出最多 2 个不同的 followup_queries；否则标记 sufficient。"
            )),
            HumanMessage(content=json.dumps({
                "question": state["question"],
                "current_query": state["search_query"],
                "evidence": state.get("retrieved_chunks", []),
                "prior_history": state.get("reasoning_logs", []),
            }, ensure_ascii=False)),
        ]
        analysis = _as_model(ResearchAnalysis, llm.with_structured_output(ResearchAnalysis).invoke(prompt))
        logs = state.get("reasoning_logs", []) + [f"证据分析：{analysis.evidence_summary}"]
        if analysis.conflict_notes:
            logs.extend(f"冲突线索：{note}" for note in analysis.conflict_notes)
        followups = [item.strip() for item in analysis.followup_queries if item.strip()]
        can_continue = bool(followups) and state.get("iteration", 0) < state.get("max_iterations", max_iterations)
        if not analysis.sufficient and can_continue:
            next_query = followups[0]
            if next_query.casefold() == state.get("search_query", "").casefold():
                can_continue = False
            else:
                logs.append(f"证据暂不足，改写检索式并继续：{next_query}")
                return {
                    "analysis": analysis,
                    "search_query": next_query,
                    "queries": state.get("queries", []) + [next_query],
                    "reasoning_logs": logs,
                    "route": "retrieve",
                }
        if state.get("iteration", 0) >= state.get("max_iterations", max_iterations) and not analysis.sufficient:
            logs.append("达到最大检索轮数，停止循环并按现有证据回答或说明不足。")
        return {"analysis": analysis, "reasoning_logs": logs, "route": "answer"}

    def answer(state: AgentState) -> dict[str, Any]:
        """生成带引用的综述；引用校验失败时回退为带来源的片段摘录。"""
        plan: ResearchPlan = state["plan"]
        if plan.clarification_question:
            return {"final_answer": plan.clarification_question, "citations": []}

        chunks = state.get("retrieved_chunks", [])
        if plan.requires_retrieval and not chunks:
            final_answer = (
                "当前知识库没有检索到可支持该问题的文献片段，因此我不会生成未经来源支持的综述结论。"
                "请先导入相关 PDF/Markdown 文献，或放宽文献范围、年份筛选后重试。"
            )
            return {"final_answer": final_answer, "citations": []}

        citations_by_id: dict[str, Citation] = {}
        for chunk in chunks:
            citation = _make_citation(chunk)
            citations_by_id[citation.citation_id] = citation
        if not citations_by_id:
            return {
                "final_answer": "请告诉我你的研究主题、关注对象或希望比较的维度，我再帮你规划文献检索。",
                "citations": [],
            }

        prompt = [
            SystemMessage(content=(
                "根据给定文献片段撰写中文综述。所有实质性陈述都必须放入结构化主张，并至少引用一个提供的 citation_id；只能使用这些 ID。"
                "严格区分文献观点（忠实转述单篇文献）、综合推断（跨文献归纳）、观点冲突（并列冲突证据）、局限说明（有来源支持的局限）。"
                "不得输出无引用的摘要或前言，不得把相关性写成因果，不得补造实验数据/文献细节；每项冲突和局限也必须附引用。"
            )),
            HumanMessage(content=json.dumps({
                "question": state["question"],
                "evidence": [
                    {"citation_id": _make_citation(chunk).citation_id, "excerpt": chunk["page_content"], "metadata": chunk["metadata"]}
                    for chunk in chunks
                ],
                "analysis": state.get("analysis").model_dump() if state.get("analysis") else None,
            }, ensure_ascii=False)),
        ]
        try:
            draft = _as_model(AnswerDraft, llm.with_structured_output(AnswerDraft).invoke(prompt))
            all_claims = draft.claims + draft.conflicts + draft.limitations
            for claim in all_claims:
                if not claim.citation_ids or any(item not in citations_by_id for item in claim.citation_ids):
                    raise ValueError("答案中存在缺失或无效的引用 ID")
            used_ids = {item for claim in all_claims for item in claim.citation_ids}
            citations = [citations_by_id[item] for item in sorted(used_ids)]
            final_answer = _render_answer(draft, citations)
        except Exception as exc:
            logger.warning("答案引用校验未通过，回退到带来源的片段摘录: %s", exc)
            citations = list(citations_by_id.values())[:5]
            lines = ["模型答案未通过引用校验，以下仅列出检索到的原文片段，不作无来源推断："]
            for citation in citations:
                lines.append(f"- [文献观点] {citation.excerpt[:420]} [{citation.citation_id}]")
                year = f", {citation.year}" if citation.year else ""
                lines.append(f"  来源：{citation.authors}{year}. {citation.title}. ({citation.source})")
            final_answer = "\n".join(lines)
        return {"final_answer": final_answer, "citations": [item.model_dump() for item in citations]}

    def output(state: AgentState) -> dict[str, Any]:
        """把本轮问答追加到会话历史，保留最近若干轮。"""
        history = state.get("conversation_history", [])
        updated = history + [{"question": state["question"], "answer": state.get("final_answer", "")}]
        return {"conversation_history": updated[-12:]}

    def route(state: AgentState) -> str:
        """根据 state.route 决定下一节点是继续检索还是生成回答。"""
        return "retrieve" if state.get("route") == "retrieve" else "answer"

    builder = StateGraph(AgentState)
    builder.add_node("intent_parse", parse_intent)
    builder.add_node("tool_decision", decide_tool)
    builder.add_node("literature_retrieval", retrieve)
    builder.add_node("reason_and_check", analyze_evidence)
    builder.add_node("answer", answer)
    builder.add_node("output", output)
    builder.add_edge(START, "intent_parse")
    builder.add_edge("intent_parse", "tool_decision")
    builder.add_conditional_edges("tool_decision", route, {"retrieve": "literature_retrieval", "answer": "answer"})
    builder.add_edge("literature_retrieval", "reason_and_check")
    builder.add_conditional_edges("reason_and_check", route, {"retrieve": "literature_retrieval", "answer": "answer"})
    builder.add_edge("answer", "output")
    builder.add_edge("output", END)
    return builder.compile(checkpointer=MemorySaver())
