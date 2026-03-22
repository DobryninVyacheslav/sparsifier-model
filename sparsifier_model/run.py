import argparse
import itertools
import time
from transformers import AutoTokenizer
import wandb

from spar_k_means_bert.util.eval import eval_with_dot_score_function
import torch
from tqdm.autonotebook import trange

from common.bench import run_bench, calc_stats
from common.datasets import load_dataset
from common.in_memory_index import InMemoryInvertedIndex
from common.lucene_inverted_index import LuceneInvertedIndex, func_to_bench, to_terms_and_scores
from sparsifier_model.config import Config, ModelType
from sparsifier_model.k_sparse.model import Autoencoder
from sparsifier_model.util.model import build_encode_sparse_fun


class InMemoryIndexRunner:
    def __init__(self, encode_fun, dataset=None, docs_number=None):
        self.encode = encode_fun
        self.index_path = "./runs/sparsifier_model/in_memoty_inverted_index"
        corpus, self.queries, self.qrels = load_dataset(
            dataset=dataset, length=docs_number
        )
        self.corpus = {
            doc_id: (doc["title"] + " " + doc["text"]).strip()
            for doc_id, doc in corpus.items()
        }
        self.inverted_index = InMemoryInvertedIndex()

    def index(self, batch_size=300):
        corpus_items = self.corpus.items()
        for start_idx in trange(0, len(self.corpus), batch_size, desc="docs"):
            batch = tuple(
                itertools.islice(corpus_items, start_idx, start_idx + batch_size)
            )
            doc_ids, docs = list(zip(*batch))
            emb_batch = self.encode(docs)
            for i in range(len(emb_batch)):
                self.inverted_index.add(doc_ids[i], emb_batch[i])

    def search(self, top_k=10):
        results = {}

        query_ids = list(self.queries.keys())
        for query_id in query_ids:
            query_emb = self.encode([self.queries[query_id]])[0]
            hits = self.inverted_index.search(query_emb, top_k)

            query_result = {}
            for hit in hits:
                query_result[hit[0]] = hit[1]
            results[query_id] = query_result
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-or-bench', type=str, default='eval', help='eval or bench (default eval)')
    parser.add_argument('-l', '--dataset-length', type=int, default=None, help='Dataset length (default None, all dataset)')
    args = parser.parse_args()
    print(f"Params: {args}")

    config = Config(model_type=ModelType.K_SPARSE)
    print("Device:", config.device, torch.cuda.is_available())
    print("Torch:", torch.__version__)

    wandb_model_name = "model-c2rfcu3i:v0"
    run = wandb.init()
    artifact = run.use_artifact(
        f"vector-search/{config.project}/{wandb_model_name}", type="model"
    )
    artifact_dir = artifact.download()
    model = Autoencoder.load_from_checkpoint(
        f"artifacts/{wandb_model_name}/model.ckpt"
    ).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_model_id, use_fast=True)
    model.eval()
    model.freeze()
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False
    encode_sparse_from_docs = build_encode_sparse_fun(
        config=config, tokenizer=tokenizer, model=model, threshold=None
    )
    print("Vector shape:", encode_sparse_from_docs("test").shape)
    print("Number of nonzero", torch.count_nonzero(encode_sparse_from_docs("test")))

    corpus, queries, qrels = load_dataset(dataset="msmarco", split="dev", length=args.dataset_length)
    print(f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}")
    inverted_index = LuceneInvertedIndex(index_path="./runs/sparsifier_model/lucene_inverted_index")
    if inverted_index.size() == 0:
        batch_size=300
        corpus = {
            doc_id: (doc["title"] + " " + doc["text"]).strip()
            for doc_id, doc in corpus.items()
        }
        inverted_index.index_all(corpus, batch_size, encode_sparse_from_docs)
    print("Inverted index size:", inverted_index.size())

    if args.eval_or_bench == "eval":
        start = time.time()
        search_results = inverted_index.search(
            queries=queries,
            sparse_vector_calculator=lambda query: to_terms_and_scores(
                encode_sparse_from_docs(query)
            ),
            top_k=1000,
        )
        print("Search time:", time.time() - start)

        ndcg, _map, recall, precision, mrr = eval_with_dot_score_function(
            qrels=qrels, results=search_results
        )
        print(ndcg, _map, recall, precision, mrr)
    else:
        reader, searcher = inverted_index.get_reader_and_searcher()
        try:
            # Warmup
            warmup_query = "warmup benchmark query for measuring search latency"
            run_bench(
                func_to_bench=lambda: func_to_bench(
                    inverted_index, searcher, warmup_query, encode_sparse_from_docs,
                ),
                warmup=1000,
                repeats=10,
            )
            # Bench
            queries = list(queries.values())
            repeats = max(2000, len(queries)) // len(queries)
            samples = []
            for query in queries:
                query_samples = run_bench(
                    func_to_bench=lambda: func_to_bench(
                        inverted_index, searcher, query, encode_sparse_from_docs,
                    ),
                    warmup=0,
                    repeats=repeats,
                )
                samples.extend(query_samples)
            calc_stats("Bench results", samples)
        finally:
            reader.close()
