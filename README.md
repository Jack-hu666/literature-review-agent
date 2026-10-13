# Literature Review Agent｜文献综述智能助手
基于LangGraph、Chroma开发的科研文献综述Agent，支持PDF与Markdown文献导入，构建本地知识库。

## ✨ 项目简介
面向科研人员的本地文献综述Agent。导入PDF、Markdown论文，文本切片后存入Chroma向量库；利用LangGraph编排多Agent，提取文献观点，自动生成综述初稿，本地运行保障隐私。

## 🛠️ 技术栈
- Python
- LangGraph
- Chroma
- LangChain
- PyPDF

## 📁 项目结构

```
├── src/
│   └── lit_review_agent/   # Agent、图状态、检索、解析与 CLI
├── scripts/                # 演示 PDF 生成器
├── tests/                  # 离线自动化测试
├── demo_papers/            # 演示文献（PDF/Markdown）
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## 🔧 环境准备

1. Python 3.10 或更高版本。
2. 复制 `.env.example` 为 `.env`，填写模型服务商 API Key：使用 OpenAI 时设置 `OPENAI_API_KEY`；使用兼容服务时同时设置 `OPENAI_BASE_URL` 与 `OPENAI_MODEL`。
3. 安装依赖并安装本包（可编辑模式）：

```bash
pip install -e .
```

## 🚀 快速启动

```bash
python -m lit_review_agent ingest demo_papers              # 导入演示文献到本地向量库
python -m lit_review_agent ask "比较两篇演示研究的结论"      # 一次性提问
python -m lit_review_agent chat                             # 多轮研究对话，输入 exit 退出
```

## 📖 使用说明

| 命令 | 说明 |
| --- | --- |
| `ingest <文件或目录>` | 解析 PDF/Markdown 文献，切片后写入 Chroma 向量库 |
| `ask "<问题>"` | 一次性生成带引用的综述；可用 `--source`、`--from-year`、`--to-year` 过滤来源/年份 |
| `chat` | 进入多轮对话；`--session ID` 指定会话，`--source` 等范围参数持续生效 |

## 🧭 工作流程

`意图解析 → 混合检索（Chroma 语义 + BM25 关键词，RRF 融合）→ 证据分析 → 综述生成与引用校验`。每条观点都绑定可核验的引用 ID；证据不足时会改写检索式继续检索，或明确说明无法支持结论，不编造综述。

## 🧩 功能亮点

- 支持 PDF/Markdown 文献解析
- Chroma 本地向量检索
- LangGraph 多 Agent 协同生成综述
- 模块化，易于扩展
