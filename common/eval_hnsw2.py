from beir.retrieval.evaluation import EvaluateRetrieval
import faiss
import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer

from common.datasets import load_dataset
from common.encode_dense_fun_builder import build_encode_dense_fun
from common.model import load_model

backbone_model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(backbone_model_id, use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(backbone_model_id, device)
encode_dense = build_encode_dense_fun(tokenizer=tokenizer, model=model, device=device)

corpus, queries, qrels = load_dataset()
hnsw_index = faiss.IndexHNSWFlat(model.config.hidden_size, 32)
hnsw_index.hnsw.efConstruction = 40
hnsw_index.hnsw.efSearch = 16
faiss_idx_to_doc_id = {}
print(
    f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}"
)
sep = " "
for doc_id, doc in tqdm.tqdm(iterable=corpus.items(), desc="build_hnsw"):
    doc = (doc["title"] + sep + doc["text"]).strip()
    doc_emb = encode_dense(doc).cpu()[0]
    hnsw_index.add(np.array([doc_emb]))
    faiss_idx_to_doc_id[hnsw_index.ntotal - 1] = doc_id
print("HNSW index size", hnsw_index.ntotal)
results = {}
unknown_doc_count = 0
for query_id, query in tqdm.tqdm(iterable=queries.items(), desc="search"):
    query_emb = encode_dense(query).cpu()
    D, I = hnsw_index.search(query_emb, 1000)
    faiss_ids = I.flatten()
    doc_ids = [faiss_idx_to_doc_id.get(id, -1) for id in faiss_ids]
    query_result = {}
    for i in range(len(doc_ids)):
        doc_id = doc_ids[i]
        if doc_id != -1:
            query_result[doc_ids[i]] = -float(D[0][i])
        else:
            unknown_doc_count += 1
    results[query_id] = query_result

print("Hm count", unknown_doc_count)
faiss.write_index(hnsw_index, "/home/slava/Developer/SparKBERT/scifact.hnsw.faiss")
##### Evaluate your retrieval using NDCG@k, MAP@K ...
retriever = EvaluateRetrieval(score_function="dot")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
print(ndcg)
print(mrr)
print(_map)
print(recall)
print(precision)
