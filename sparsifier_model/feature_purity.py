import argparse
import heapq
import json
import re
from collections import Counter
from pathlib import Path

import torch
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer

from common.datasets import load_dataset
from sparsifier_model.config import Config, ModelType
from sparsifier_model.k_sparse.model import Autoencoder
from sparsifier_model.util.model import build_encode_sparse_fun


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def _truncate(text: str, max_chars: int) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    if not truncated:
        truncated = text[:max_chars]
    return truncated + "..."


def _tokenize(text: str) -> list[str]:
    tokens = [tok.lower() for tok in WORD_RE.findall(text)]
    return [tok for tok in tokens if len(tok) >= 3 and tok not in STOPWORDS]


def _compute_purity(texts: list[str], top_n: int) -> tuple[float, list[str], float]:
    if not texts:
        return 0.0, [], 0.0
    doc_freq = Counter()
    token_sets = []
    for text in texts:
        tokens = set(_tokenize(text))
        token_sets.append(tokens)
        doc_freq.update(tokens)
    if not doc_freq:
        return 0.0, [], 0.0
    top_keywords = [term for term, _ in doc_freq.most_common(top_n)]
    top_keyword = top_keywords[0]
    purity = doc_freq[top_keyword] / len(texts)
    coverage = sum(1 for tokens in token_sets if tokens & set(top_keywords)) / len(texts)
    return purity, top_keywords, coverage


def _load_docs_from_file(path: Path, text_field: str, max_docs: int | None):
    docs = []
    if path.suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data.get(text_field) or data.get("text") or data.get("contents") or ""
                if not text:
                    continue
                doc_id = data.get("id") or data.get("_id") or str(len(docs))
                docs.append({"id": str(doc_id), "text": text})
                if max_docs and len(docs) >= max_docs:
                    break
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                docs.append({"id": str(len(docs)), "text": text})
                if max_docs and len(docs) >= max_docs:
                    break
    return docs


def _load_docs_from_dataset(dataset: str, split: str, max_docs: int | None):
    local_path = Path("datasets") / dataset
    if local_path.exists():
        corpus, _, _ = GenericDataLoader(str(local_path)).load(split=split)
    else:
        corpus, _, _ = load_dataset(dataset=dataset, split=split, length=max_docs)
    docs = []
    for doc_id, doc in corpus.items():
        text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
        if not text:
            continue
        docs.append({"id": str(doc_id), "text": text})
        if max_docs and len(docs) >= max_docs:
            break
    return docs


def _iter_batches(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _load_model(checkpoint: Path, device: torch.device):
    model = Autoencoder.load_from_checkpoint(str(checkpoint))
    model.to(device)
    model.eval()
    model.freeze()
    config = model.hparams.get("config", Config(model_type=ModelType.K_SPARSE))
    config.device = device
    return model, config


def _parse_feature_ids(feature_ids: str | None, latent_dim: int):
    if not feature_ids:
        return None
    ids = []
    for part in feature_ids.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= latent_dim:
            raise ValueError(f"Feature id {idx} out of range [0, {latent_dim - 1}]")
        ids.append(idx)
    mask = [False] * latent_dim
    for idx in ids:
        mask[idx] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Surface top-activating docs per sparse feature and a simple purity score."
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--docs-file", type=str, help="Path to JSONL or text file of documents")
    sources.add_argument("--dataset", type=str, help="BEIR dataset name (uses ./datasets/<name> if present)")
    parser.add_argument("--dataset-split", type=str, default="dev", help="Dataset split")
    parser.add_argument("--text-field", type=str, default="text", help="JSONL field containing text")
    parser.add_argument("--max-docs", type=int, default=2000, help="Max number of docs to process")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding")
    parser.add_argument("--top-k", type=int, default=20, help="Top docs to keep per feature")
    parser.add_argument("--top-keywords", type=int, default=8, help="Keywords to report per feature")
    parser.add_argument("--min-activation", type=float, default=0.0, help="Minimum activation to keep")
    parser.add_argument("--feature-ids", type=str, default=None, help="Comma-separated feature ids to analyze")
    parser.add_argument("--summary-top", type=int, default=20, help="Number of features to print in summary")
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="runs/feature_purity.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g., cpu, cuda)")
    args = parser.parse_args()

    docs = []
    if args.docs_file:
        docs = _load_docs_from_file(Path(args.docs_file), args.text_field, args.max_docs)
    else:
        docs = _load_docs_from_dataset(args.dataset, args.dataset_split, args.max_docs)
    if not docs:
        raise RuntimeError("No documents loaded.")

    device = torch.device(args.device) if args.device else Config(ModelType.K_SPARSE).device
    model, config = _load_model(Path(args.checkpoint), device)
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_model_id, use_fast=True)
    encode_fun = build_encode_sparse_fun(
        config=config, tokenizer=tokenizer, model=model, threshold=None
    )

    latent_dim = model.latent_bias.shape[0]
    feature_mask = _parse_feature_ids(args.feature_ids, latent_dim)
    topk_heaps = [[] for _ in range(latent_dim)]
    activation_counts = [0] * latent_dim

    with torch.no_grad():
        for batch in _iter_batches(docs, args.batch_size):
            texts = [item["text"] for item in batch]
            doc_ids = [item["id"] for item in batch]
            vectors = encode_fun(texts).detach().to("cpu")
            for row_idx in range(vectors.shape[0]):
                vec = vectors[row_idx]
                nonzero = torch.nonzero(vec, as_tuple=True)[0]
                if args.min_activation > 0:
                    nonzero = nonzero[vec[nonzero] > args.min_activation]
                for feature_idx in nonzero.tolist():
                    if feature_mask and not feature_mask[feature_idx]:
                        continue
                    score = float(vec[feature_idx])
                    activation_counts[feature_idx] += 1
                    heap = topk_heaps[feature_idx]
                    entry = (score, doc_ids[row_idx], texts[row_idx])
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, entry)
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, entry)

    results = []
    for feature_idx, heap in enumerate(topk_heaps):
        if feature_mask and not feature_mask[feature_idx]:
            continue
        if not heap:
            continue
        docs_sorted = sorted(heap, key=lambda item: item[0], reverse=True)
        doc_texts = [item[2] for item in docs_sorted]
        purity, keywords, coverage = _compute_purity(doc_texts, args.top_keywords)
        results.append(
            {
                "feature_id": feature_idx,
                "purity": round(purity, 4),
                "keyword_coverage": round(coverage, 4),
                "activation_count": activation_counts[feature_idx],
                "top_keywords": keywords,
                "docs": [
                    {
                        "doc_id": doc_id,
                        "score": round(score, 6),
                        "text": _truncate(text, 240),
                    }
                    for score, doc_id, text in docs_sorted
                ],
            }
        )

    results.sort(
        key=lambda item: (item["purity"], item["activation_count"]), reverse=True
    )
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    print(f"Processed docs: {len(docs)}")
    print(f"Features with activations: {len(results)}")
    print("Top features by purity:")
    for item in results[: args.summary_top]:
        keywords = item["top_keywords"]
        top_keyword = keywords[0] if keywords else "n/a"
        print(
            f"  feature {item['feature_id']}: purity={item['purity']:.2f}, "
            f"activated={item['activation_count']}, top_keyword={top_keyword}"
        )


if __name__ == "__main__":
    main()
