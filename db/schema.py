import duckdb
import os
from config.settings import settings

def get_connection():
    os.makedirs(os.path.dirname(settings.duckdb_path), exist_ok=True)
    return duckdb.connect(settings.duckdb_path)

def init_schema():
    con = get_connection()

    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id   VARCHAR PRIMARY KEY,
            faiss_id   INTEGER UNIQUE,
            doc_name   VARCHAR,
            page_num   INTEGER,
            chunk_text TEXT,
            created_at TIMESTAMP DEFAULT now()
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS verdicts (
            job_id           VARCHAR PRIMARY KEY,
            doc_name         VARCHAR,
            doc_hash         VARCHAR,
            risk_score       FLOAT,
            finance_score    FLOAT,
            compliance_score FLOAT,
            overall_score    FLOAT,
            debate_log       JSON,
            final_verdict    TEXT,
            audit_hash       VARCHAR,
            created_at       TIMESTAMP DEFAULT now()
        )
    """)

    con.close()
    print("Schema ready.")

if __name__ == "__main__":
    init_schema()