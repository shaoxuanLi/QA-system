# Ablation on dev.jsonl

_N = 1000_

| Run | Reranker | top_k_retrieve | top_k_final | EM | F1 | Time(s) |
|---|---|---|---|---|---|---|
| bm25-only-top1 | off | 1 | 1 | 11.30 | 17.51 | 180.6 |
| bm25-only-top3 | off | 3 | 3 | 10.20 | 15.65 | 676.7 |
| bm25-only-top5 | off | 5 | 5 | 9.40 | 13.90 | 1237.8 |
| bm25+rerank-top1 | on | 20 | 1 | 13.70 | 19.61 | 950.8 |
| bm25+rerank-top3 | on | 20 | 3 | 15.50 | 20.61 | 1297.7 |
| bm25+rerank-top5 | on | 20 | 5 | 10.80 | 16.09 | 1677.6 |
| bm25+rerank-recall50 | on | 50 | 5 | 11.90 | 17.01 | 2722.9 |
| bm25+rerank-recall100 | on | 100 | 5 | 12.60 | 17.88 | 4425.3 |
