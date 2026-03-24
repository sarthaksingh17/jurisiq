import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from config.settings import settings
from db.schema import get_connection


@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def load_index():
    return faiss.read_index(settings.faiss_index_path)


def similarity_search(query: str, top_k: int = 5) -> list:
    model = load_model()
    index = load_index()

    vec = model.encode([query], normalize_embeddings=True)
    vec = np.array(vec, dtype="float32")

    scores, faiss_ids = index.search(vec, top_k)
    ids = faiss_ids[0].tolist()
    scs = scores[0].tolist()

    if not ids or ids[0] == -1:
        return []

    con = get_connection()
    placeholders = ",".join("?" * len(ids))
    rows = con.execute(f"""
        SELECT faiss_id, chunk_text, doc_name, page_num
        FROM chunks
        WHERE faiss_id IN ({placeholders})
    """, ids).fetchall()
    con.close()

    score_map = {fid: sc for fid, sc in zip(ids, scs)}
    results = []
    for faiss_id, chunk_text, doc_name, page_num in rows:
        results.append({
            "chunk_text": chunk_text,
            "doc_name":   doc_name,
            "page_num":   page_num,
            "score":      round(float(score_map.get(faiss_id, 0)), 4),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)