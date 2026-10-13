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
├── main.py
├── agent/
├── loader/
├── vector_store/
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 快速启动
```bash
pip install -r requirements.txt
python main.py
```

## 🧩 功能亮点

- 支持 PDF/Markdown 文献解析
- Chroma 本地向量检索
- LangGraph 多 Agent 协同生成综述
- 模块化，易于扩展
