# Title: Hybrid Retrieval for Small Research Collections: A Demonstration Study
Authors: Demo Research Group
Year: 2022
Abstract: This fictional demonstration paper compares dense semantic retrieval and BM25 keyword retrieval on a small, manually curated research collection. The synthetic experiment suggests that dense retrieval finds relevant passages when query wording differs from the paper wording, while BM25 is more reliable for exact technical terms. The document is intentionally illustrative and reports no real-world measurements.

## Methods
The demonstration uses a small synthetic corpus of research summaries. Questions are written in both direct keyword form and paraphrased form. Dense retrieval uses multilingual sentence embeddings. Keyword retrieval uses BM25 term statistics. The comparison is qualitative and is not a benchmark study.

## Findings
For paraphrased questions, semantic similarity can retrieve passages that share meaning but few exact words with the question. For rare model names and exact dataset identifiers, lexical overlap remains useful. The paper therefore recommends combining the two candidate lists and checking the original passage before summarizing it.

## Limitations
All examples and observations in this document are fictional and prepared solely for software demonstration. No measured recall, statistical significance, or real publication claim is made.
