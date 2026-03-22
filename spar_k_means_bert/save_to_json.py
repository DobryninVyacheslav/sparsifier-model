import pickle
from pathlib import Path
import json
import os


def load_faiss_idx_to_token():
    print(os.getcwd())
    faiss_idx_to_token_file_name = "./faiss_idx_to_token.pickle"
    with open(faiss_idx_to_token_file_name, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    faiss_idx_to_token = load_faiss_idx_to_token()
    faiss_idx_to_token_str_keys = {str(k): v for k, v in faiss_idx_to_token.items()}
    out_path = Path("./faiss_idx_to_token.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            faiss_idx_to_token_str_keys,
            f,
            indent=2,
        )
