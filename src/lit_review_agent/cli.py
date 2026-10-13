import argparse
import logging
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from lit_review_agent.config import Settings
from lit_review_agent.graph import build_graph
from lit_review_agent.ingestion import load_documents
from lit_review_agent.logging_config import configure_logging
from lit_review_agent.retrieval import HybridRetriever
from lit_review_agent.schemas import ResearchScope

logger = logging.getLogger(__name__)


def create_vector_store(settings: Settings) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    settings.chroma_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="literature_chunks",
        persist_directory=str(settings.chroma_directory),
        embedding_function=embeddings,
    )


def ingest(path: Path, settings: Settings) -> None:
    if not path.exists():
        raise SystemExit(f"输入路径不存在：{path}")
    documents, skipped = load_documents(path, settings)
    if not documents:
        raise SystemExit("没有解析到文献文本。请检查 PDF 是否可复制文本，或文件扩展名是否为 PDF/Markdown。")
    store = create_vector_store(settings)
    store.add_documents(documents, ids=[item.metadata["chunk_id"] for item in documents])
    papers = {item.metadata["paper_id"] for item in documents}
    print(f"已导入 {len(papers)} 篇文献、{len(documents)} 个片段；跳过 {skipped} 个文件。")
    print(f"Chroma 持久化目录：{settings.chroma_directory}")


def create_agent(settings: Settings):
    api_key = settings.require_api_key()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
    retriever = HybridRetriever(create_vector_store(settings), top_k=settings.retrieval_top_k)
    return build_graph(llm, retriever, max_iterations=settings.max_search_iterations)


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 RAG + ReAct + LangGraph 的文献综述助手")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="导入 PDF/Markdown 文献")
    ingest_parser.add_argument("path", type=Path, help="文献文件或目录")
    ask_parser = commands.add_parser("ask", help="提出一个研究问题")
    ask_parser.add_argument("question", help="研究问题")
    ask_parser.add_argument("--session", default=None, help="对话会话 ID")
    ask_parser.add_argument("--source", default=None, help="限制文献名、作者或来源路径")
    ask_parser.add_argument("--from-year", type=int, default=None, help="起始年份，包含")
    ask_parser.add_argument("--to-year", type=int, default=None, help="截止年份，包含")
    chat_parser = commands.add_parser("chat", help="进入多轮研究对话")
    chat_parser.add_argument("--session", default=None, help="会话 ID；默认自动生成")
    chat_parser.add_argument("--source", default=None, help="限制文献名、作者或来源路径")
    chat_parser.add_argument("--from-year", type=int, default=None, help="起始年份，包含")
    chat_parser.add_argument("--to-year", type=int, default=None, help="截止年份，包含")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level)
    if args.command == "ingest":
        ingest(args.path, settings)
        return

    try:
        session_id = args.session or str(uuid.uuid4())
        scope = ResearchScope(source_filter=args.source, from_year=args.from_year, to_year=args.to_year)
        graph = create_agent(settings)
    except Exception as exc:
        raise SystemExit(f"Agent 初始化失败：{exc}") from exc

    if args.command == "ask":
        result = graph.invoke(
            {"question": args.question, "scope": scope},
            config={"configurable": {"thread_id": session_id}},
        )
        print(result.get("final_answer", "本轮没有生成结果。"))
        return

    print(f"文献研究助手已启动（session={session_id}）。输入 exit 退出。")
    print("本轮范围参数会持续应用；如需调整，请退出后用新的 --source/年份参数重新启动。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if question.casefold() in {"exit", "quit"}:
            print("已退出。")
            break
        if not question:
            continue
        try:
            result = graph.invoke(
                {"question": question, "scope": scope},
                config={"configurable": {"thread_id": session_id}},
            )
            print(f"\n{result.get('final_answer', '本轮没有生成结果。')}")
        except Exception:
            logger.exception("本轮研究任务失败")
            print("本轮处理失败，请检查日志、API 配置和模型服务状态后重试。")


if __name__ == "__main__":
    main()
