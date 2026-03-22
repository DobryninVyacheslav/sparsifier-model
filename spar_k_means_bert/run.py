import time
import os
import pickle
from collections import defaultdict

import faiss
import numpy as np
import sklearn.cluster
import torch
import tqdm
from transformers import AutoTokenizer

from common.encode_dense_fun_builder import build_encode_dense_fun
from common.lucene_inverted_index import LuceneInvertedIndex
from common.model import load_model
from spar_k_means_bert.args import get_args
from spar_k_means_bert.dataset import get_dataset, get_dataloader
from spar_k_means_bert.in_memory_inverted_index import InMemoryInvertedIndex
from spar_k_means_bert.util.encode import encode_to_token_embs
from spar_k_means_bert.util.eval import eval_with_dot_score_function

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)


def get_contextualized_embs(doc_id_to_embs, token, doc_id):
    input_ids, _ = dataset.get_by_doc_id(doc_id)
    embs = doc_id_to_embs[doc_id]
    idxs = torch.nonzero(input_ids == token, as_tuple=True)
    return embs[idxs]


def build_token_to_doc_ids():
    token_to_doc_ids = defaultdict(set)
    skip_tokens = {0, 101, 102}
    for doc_ids, token_ids_batch, _ in tqdm.tqdm(iterable=dataloader, desc="build_token_to_doc_ids"):
        for idx, doc_id in enumerate(doc_ids):
            for token_id in token_ids_batch[idx]:
                token_id = token_id.item()
                if token_id not in skip_tokens:
                    token_to_doc_ids[token_id].add(doc_id)
    return token_to_doc_ids


def build_doc_id_to_embs():
    doc_id_to_embs = {}
    for doc_ids, token_ids_batch, attention_mask in tqdm.tqdm(iterable=dataloader, desc="encode_to_token_embs"):
        embs = encode_to_token_embs(model=model, input_ids=token_ids_batch, attention_mask=attention_mask)
        embs = embs.cpu()
        for idx, doc_id in enumerate(doc_ids):
            doc_id_to_embs[doc_id] = embs[idx].unsqueeze(0)
    return doc_id_to_embs


def train_vector_dictionary():
    hnsw_file_name = f"{args.base_path}hnsw.index"
    faiss_idx_to_token_file_name = f"{args.base_path}faiss_idx_to_token.pickle"

    if args.use_cache and os.path.isfile(hnsw_file_name) and os.path.isfile(faiss_idx_to_token_file_name):
        with open(faiss_idx_to_token_file_name, "rb") as f:
            return faiss.read_index(hnsw_file_name), pickle.load(f)

    if os.path.isfile(hnsw_file_name):
        os.remove(hnsw_file_name)
    if os.path.isfile(faiss_idx_to_token_file_name):
        os.remove(faiss_idx_to_token_file_name)

    token_to_doc_ids = build_token_to_doc_ids()
    doc_id_to_embs = build_doc_id_to_embs()
    hnsw_index = faiss.IndexHNSWFlat(model.config.hidden_size, args.hnsw_M)
    hnsw_index.hnsw.efConstruction = args.hnsw_ef_construction
    faiss_idx_to_token = {}

    for token, doc_ids in tqdm.tqdm(iterable=token_to_doc_ids.items(), desc="train_vector_dictionary"):
        emb_batches = []
        for doc_id in doc_ids:
            emb_batch = get_contextualized_embs(doc_id_to_embs, token, doc_id)
            emb_batches.append(emb_batch.cpu().detach().numpy())
        embs = np.concatenate(emb_batches)
        kmeans = sklearn.cluster.KMeans(
            n_clusters=args.kmeans_n_clusters if len(embs) > args.kmeans_n_clusters else len(embs),
            init='k-means++',
            n_init='auto')
        kmeans.fit(embs)
        for i, centroid in enumerate(kmeans.cluster_centers_):
            hnsw_index.add(np.array([centroid]))
            faiss_idx_to_token[hnsw_index.ntotal - 1] = f"{token}_{i}"

    faiss.write_index(hnsw_index, hnsw_file_name)
    with open(faiss_idx_to_token_file_name, "wb") as f:
        pickle.dump(faiss_idx_to_token, f)
    print(f"Constructed HNSW levels: {np.bincount(faiss.vector_to_array(hnsw_index.hnsw.levels))}")
    return hnsw_index, faiss_idx_to_token


def build_inverted_index():
    if args.in_memory_index:
        inverted_index = InMemoryInvertedIndex(args.base_path, args.use_cache)
    else:
        index_path = f"{args.base_path}runs/spar_k_means_bert/lucene_inverted_index"
        inverted_index = LuceneInvertedIndex(index_path, threshold)
        if not args.use_cache:
            inverted_index.delete_index()
    if inverted_index.size():
        return inverted_index
    for doc_ids, token_ids_batch, attention_mask in tqdm.tqdm(iterable=dataloader, desc="build_inverted_index"):
        embs = encode_to_token_embs(model=model, input_ids=token_ids_batch, attention_mask=attention_mask)
        embs = embs.cpu()
        for idx, doc_id in enumerate(doc_ids):
            contextualized_embs = embs[idx]
            _, I = hnsw_index.search(contextualized_embs, args.index_n_neighbors)
            assert len(I) == len(contextualized_embs)
            faiss_ids = np.unique(I.flatten())  # this help to remove token repetition
            token_and_cluster_id_list = [faiss_idx_to_token[id] for id in faiss_ids]
            centroids = torch.from_numpy(hnsw_index.reconstruct_batch(faiss_ids))
            scores = torch.max(contextualized_embs @ centroids.T, dim=0).values  # MaxSim
            inverted_index.index(doc_id, token_and_cluster_id_list, scores)
    inverted_index.complete_indexing()
    return inverted_index


def query_tokens_calculator(query):
    tokenized_query = tokenize(query)
    contextualized_embs = encode_to_token_embs(
        model=model,
        input_ids=tokenized_query["input_ids"],
        attention_mask=tokenized_query["attention_mask"])
    contextualized_embs_np = contextualized_embs.squeeze(0).cpu().detach().numpy()
    _, I = hnsw_index.search(contextualized_embs_np, args.search_n_neighbors)
    assert len(I) == len(contextualized_embs_np)
    return list(map(lambda idx: faiss_idx_to_token[idx], np.unique(I.flatten()))), None


if __name__ == "__main__":
    args = get_args()
    # Data, tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_id, use_fast=True)
    dataset, queries, qrels = get_dataset(tokenize=tokenize, args=args)
    dataloader = get_dataloader(dataset=dataset, args=args, pad_token_id=tokenizer.pad_token_id)
    model = load_model(model_id=args.backbone_model_id, device=DEVICE)
    encode_dense = build_encode_dense_fun(tokenizer=tokenizer, model=model, device=DEVICE)
    threshold = 0.8 * encode_dense("She enjoys reading books in her free time.") @ encode_dense("In her leisure hours, she likes to read novels.").T
    threshold = threshold.squeeze(0).cpu()
    print(f"Dense similarity threshold: {threshold}")
    # Indexing
    hnsw_index, faiss_idx_to_token = train_vector_dictionary()
    print("HNSW index size: ", hnsw_index.ntotal)
    if args.train_hnsw_only:
        print("HNSW index is trained")
        exit(0)  # TODO: extract to function and use return
    hnsw_index.hnsw.efSearch = args.hnsw_ef_search
    inverted_index = build_inverted_index()
    # Retrieval
    start = time.time()
    results = inverted_index.search(queries, query_tokens_calculator, args.search_top_k)
    print("Search time:", time.time() - start)
    # for q_id, ids_scores in results.items():
    #     print("QUERY:", queries[q_id])
    #     top10_ids = [k for k, _ in sorted(ids_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]]
    #     for i, top10_id in enumerate(top10_ids):
    #         print(i+1, dataset.docs[dataset.doc_id_to_idx[top10_id]])
    ndcg, _map, recall, precision, mrr = eval_with_dot_score_function(qrels, results)
    print(ndcg, _map, recall, precision, mrr)
    with open("retrieval_results.txt", "a") as f:
        f.write(str(args))
        f.write("\n")
        f.write(str(ndcg))
        f.write(str(mrr))
        f.write(str(_map))
        f.write(str(recall))
        f.write(str(precision))
        f.write("\n\n")
