import hashlib
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from lit_review_agent.config import Settings
from lit_review_agent.schemas import PaperMetadata

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}


def read_source(path: Path) -> tuple[str, str | None]:
    if path.suffix.lower() in {".md", ".markdown"}:
        return path.read_text(encoding="utf-8"), None
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    pdf_title = reader.metadata.title if reader.metadata else None
    if not text.strip():
        raise ValueError("PDF 没有可提取的文本；扫描版 PDF 请先进行 OCR。")
    return text, pdf_title


def extract_metadata(text: str, source: str, pdf_title: str | None = None) -> PaperMetadata:
    lines = [line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()]

    def field_value(label: str) -> str | None:
        pattern = re.compile(rf"^{re.escape(label)}\s*:\s*(.+)$", re.IGNORECASE)
        for line in lines[:40]:
            match = pattern.match(line)
            if match:
                return match.group(1).strip()
        return None

    title = field_value("Title") or pdf_title
    if title:
        title_pattern = re.compile(r"^Title\s*:\s*(.+)$", re.IGNORECASE)
        for index, line in enumerate(lines[:40]):
            match = title_pattern.match(line)
            if not match:
                continue
            title_parts = [match.group(1).strip()]
            for following in lines[index + 1 : 40]:
                if re.match(r"^(authors?|year|abstract|keywords?)\s*:", following, re.I):
                    break
                title_parts.append(following)
            title = " ".join(title_parts)
            break
    if not title:
        title = next(
            (line for line in lines if not re.match(r"^(authors?|year|abstract)\s*:", line, re.I)),
            Path(source).stem.replace("_", " "),
        )
    authors = field_value("Authors") or field_value("Author") or "Unknown"
    year_value = field_value("Year")
    year_match = re.search(r"\b(?:19|20)\d{2}\b", year_value or " ")
    if not year_match:
        year_match = re.search(r"\b(?:19|20)\d{2}\b", " ".join(lines[:20]))
    year = int(year_match.group()) if year_match else None

    abstract = ""
    for index, line in enumerate(lines):
        match = re.match(r"^Abstract\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            parts = [match.group(1)] if match.group(1) else []
            for following in lines[index + 1 :]:
                if re.match(r"^(keywords?|\d+(?:\.\d+)*\s+|#{1,6}\s*)", following, re.I):
                    break
                if following.casefold() in {
                    "introduction", "background", "methods", "methodology", "results",
                    "findings", "discussion", "conclusion", "conclusions", "limitations",
                }:
                    break
                if re.match(r"^[A-Z][A-Za-z ]{2,30}:\s*", following):
                    break
                parts.append(following)
            abstract = " ".join(parts).strip()
            break

    paper_id = hashlib.sha1(f"{title.casefold()}|{authors.casefold()}".encode("utf-8")).hexdigest()[:16]
    return PaperMetadata(
        title=title[:500], abstract=abstract[:4000], authors=authors[:500], year=year,
        source=source, paper_id=paper_id,
    )


def load_documents(path: Path, settings: Settings) -> tuple[list[Document], int]:
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    supported = [item for item in paths if item.suffix.lower() in SUPPORTED_SUFFIXES]
    # Prefer PDF when a matching Markdown and PDF sample are both present.
    unique: dict[str, Path] = {}
    for item in supported:
        key = str(item.relative_to(path if path.is_dir() else path.parent).with_suffix("")).casefold()
        if key not in unique or item.suffix.lower() == ".pdf":
            unique[key] = item

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks: list[Document] = []
    skipped = 0
    for item in unique.values():
        try:
            text, pdf_title = read_source(item)
            relative_source = item.as_posix()
            metadata = extract_metadata(text, relative_source, pdf_title)
            for index, part in enumerate(splitter.split_text(text)):
                chunk_id = hashlib.sha1(f"{metadata.paper_id}|{index}|{part}".encode("utf-8")).hexdigest()[:20]
                chunks.append(Document(
                    page_content=part,
                    metadata={
                        "chunk_id": chunk_id,
                        "paper_id": metadata.paper_id,
                        "title": metadata.title,
                        "abstract": metadata.abstract,
                        "authors": metadata.authors,
                        "year": metadata.year or 0,
                        "source": metadata.source,
                        "chunk_index": index,
                    },
                ))
        except Exception as exc:
            logger.warning("跳过无法解析的文件 %s: %s", item, exc)
            skipped += 1
    return chunks, skipped
