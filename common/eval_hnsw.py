import os
import pathlib

from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import HNSWFaissSearch

from common.datasets import load_dataset

corpus, queries, qrels = load_dataset()
print(f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}")

model = models.SentenceBERT("msmarco-distilbert-dot-v5")
faiss_search = HNSWFaissSearch(model,
                               batch_size=128,
                               hnsw_store_n=512,
                               hnsw_ef_search=128,
                               hnsw_ef_construction=200)
#### Retrieve dense results
retriever = EvaluateRetrieval(faiss_search, score_function="dot") # or "dot" for dot-product
results = retriever.retrieve(corpus, queries)
#### Save faiss index into file or disk
prefix = "dense"      # (default value)
ext = "hnsw"             # or "pq", "hnsw"
output_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(os.path.join(output_dir, "{}.{}.faiss".format(prefix, ext))):
    faiss_search.save(output_dir=output_dir, prefix=prefix, ext=ext)
#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
print(ndcg)
print(mrr)
print(_map)
print(recall)
print(precision)
