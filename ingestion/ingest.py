import json
import uuid
import urllib.request
import numpy as np
import faiss
import os
from transformers import AutoTokenizer, AutoModel
import torch
from db.schema import get_connection, init_schema
from config.settings import settings

CUAD_JSON_URL = (
    "https://huggingface.co/datasets/theatticusproject/cuad"
    "/resolve/main/CUAD_v1/CUAD_v1.json"
)


def get_embedding(text: str, tokenizer, model) -> np.ndarray:
    encoded = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512
    )
    with torch.no_grad():
        output = model(**encoded)
    vec = output.last_hidden_state[:, 0, :].squeeze().numpy()
    vec = vec / np.linalg.norm(vec)   # normalise for cosine
    return vec.astype("float32")


def chunk_text(text: str) -> list:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + settings.chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
        start += settings.chunk_size - settings.chunk_overlap
    return chunks


def load_cuad_contracts() -> list:
    """Download CUAD SQuAD-format JSON and extract unique contracts."""
    cache_path = os.path.join(settings.corpus_dir, "CUAD_v1.json")
    os.makedirs(settings.corpus_dir, exist_ok=True)

    if not os.path.exists(cache_path):
        print(f"Downloading CUAD dataset (~40 MB)...")
        urllib.request.urlretrieve(CUAD_JSON_URL, cache_path)
        print("Download complete.")
    else:
        print("Using cached CUAD_v1.json")

    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # SQuAD format: {"data": [{"title": ..., "paragraphs": [{"context": ...}]}]}
    contracts = []
    for entry in raw.get("data", []):
        title = entry.get("title", "unknown")
        # combine all paragraph contexts into one document
        text = "\n".join(
            p["context"] for p in entry.get("paragraphs", []) if p.get("context")
        )
        if text.strip():
            contracts.append({"title": title, "text": text})

    return contracts


def ingest_cuad(limit: int = 20):
    init_schema()

    print("Loading InLegalBERT...")
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
    model     = AutoModel.from_pretrained("law-ai/InLegalBERT")
    model.eval()

    os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)

    # always start fresh to avoid dimension / id mismatches
    if os.path.exists(settings.faiss_index_path):
        os.remove(settings.faiss_index_path)

    index = faiss.IndexFlatIP(settings.embed_dim)
    print("Created new FAISS index")

    con           = get_connection()
    next_faiss_id = 0

    contracts = load_cuad_contracts()
    contracts = contracts[:limit]
    print(f"Ingesting {len(contracts)} contracts (limit={limit})")

    for i, contract in enumerate(contracts):
        title = contract["title"]
        text  = contract["text"]

        chunks = chunk_text(text)
        print(f"[{i+1}/{len(contracts)}] {title[:50]} — {len(chunks)} chunks")

        for chunk in chunks:
            vec = get_embedding(chunk, tokenizer, model)
            index.add(np.array([vec]))

            con.execute("""
                INSERT OR IGNORE INTO chunks
                    (chunk_id, faiss_id, doc_name, page_num, chunk_text)
                VALUES (?, ?, ?, ?, ?)
            """, [str(uuid.uuid4()), next_faiss_id, title, 0, chunk])

            next_faiss_id += 1

        if (i + 1) % 10 == 0:
            faiss.write_index(index, settings.faiss_index_path)
            print(f"  Checkpoint saved — {index.ntotal} vectors")

    # final save
    faiss.write_index(index, settings.faiss_index_path)
    con.close()
    print(f"Done! {index.ntotal} total vectors in index.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20,
                        help="Max contracts to ingest (default 20)")
    args = parser.parse_args()
    ingest_cuad(limit=args.limit)