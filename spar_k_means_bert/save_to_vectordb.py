import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common.model import load_model
from spar_k_means_bert.dataset import get_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models as qmodels

from spar_k_means_bert.util.encode import encode_to_token_embs

backbone_model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(backbone_model_id, use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        device
    )


if __name__ == "__main__":
    client = QdrantClient(host="vectordb.home", port=6333, timeout=60)
    collection_name = "scifact_embs_all_MiniLM_L6_v2"
    model = load_model(model_id=backbone_model_id, device=device)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=model.config.hidden_size,
                distance=Distance.DOT,
            ),
        )
    dataset, _, _ = get_dataset(tokenize=tokenize)
    dataloader = DataLoader(dataset=dataset, batch_size=128)

    point_id = 0
    for doc_ids, token_ids_batch, attention_mask in tqdm.tqdm(
        iterable=dataloader, desc="encode_to_token_embs"
    ):
        embs_batch = (
            encode_to_token_embs(
                model=model, input_ids=token_ids_batch, attention_mask=attention_mask
            )
            .cpu()
            .numpy()
        )

        points = []
        for idx, doc_id in enumerate(doc_ids):
            embs = embs_batch[idx].tolist()
            token_ids = token_ids_batch[
                idx
            ].tolist()  # если нужно сохранить сами token_ids
            for token_id, emb in zip(token_ids, embs):
                if token_id != 0:
                    points.append(
                        qmodels.PointStruct(
                            id=point_id,
                            vector=emb,
                            payload={
                                "doc_id": doc_id,
                                "token": token_id,  # здесь сохраняем токены
                            },
                        )
                    )
                    point_id += 1

        batch_size = 200
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
