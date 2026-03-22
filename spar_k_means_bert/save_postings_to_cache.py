import os
import pickle
import faiss
import numpy as np
import redis
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common.model import load_model
from spar_k_means_bert.dataset import get_dataset
from spar_k_means_bert.util.encode import encode_to_token_embs


def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        device
    )


if __name__ == "__main__":
    # Init
    backbone_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(backbone_model_id, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = redis.Redis(host="cache.home", port=16379)
    r.ping()
    model = load_model(model_id=backbone_model_id, device=device)

    # Stage 1: Get Dateset
    dataset, queries, qrels = get_dataset(tokenize=tokenize)
    dataloader = DataLoader(dataset=dataset, batch_size=128)

    # Stage 2: Get HNSW Index
    hnsw_file_name = "./hnsw.index"
    faiss_idx_to_token_file_name = "./faiss_idx_to_token.pickle"

    with open(faiss_idx_to_token_file_name, "rb") as f:
        hnsw_index, faiss_idx_to_token = (
            faiss.read_index(hnsw_file_name),
            pickle.load(f),
        )
    print("HNSW index size: ", hnsw_index.ntotal)

    # Stage 3: Build doc_id to embs
    doc_id_to_embs = {}
    for doc_ids, token_ids_batch, attention_mask in tqdm.tqdm(
        iterable=dataloader, desc="encode_to_token_embs"
    ):
        embs = encode_to_token_embs(
            model=model, input_ids=token_ids_batch, attention_mask=attention_mask
        )
        embs = embs.cpu()
        for idx, doc_id in enumerate(doc_ids):
            doc_id_to_embs[doc_id] = embs[idx].unsqueeze(0)

    # Stage 4: Build MaxSim scores
    BATCH_SIZE = 1_000
    SEP = "|"
    list_name = "postings_py"
    for doc_id, contextualized_embs in tqdm.tqdm(
        iterable=doc_id_to_embs.items(), desc="build_max_sims"
    ):
        contextualized_embs = contextualized_embs.squeeze(0)
        _, I = hnsw_index.search(contextualized_embs, 8)
        assert len(I) == len(contextualized_embs)
        faiss_ids = np.unique(I.flatten())  # this help to remove token repetition
        token_and_cluster_id_list = [faiss_idx_to_token[id] for id in faiss_ids]
        centroids = torch.from_numpy(hnsw_index.reconstruct_batch(faiss_ids))
        scores = torch.max(contextualized_embs @ centroids.T, dim=0).values  # MaxSim
        batch: list[str] = []
        pipe = r.pipeline()
        assert len(token_and_cluster_id_list) == len(scores)
        for token_and_cluster_id, score in zip(token_and_cluster_id_list, scores):
            batch.append(f"{token_and_cluster_id}{SEP}{doc_id}{SEP}{score}")

            if len(batch) == BATCH_SIZE:
                pipe.rpush(list_name, *batch)
                batch.clear()

        if batch:
            pipe.rpush(list_name, *batch)
        pipe.execute()
