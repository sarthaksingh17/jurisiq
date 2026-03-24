from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 64
    faiss_index_path: str = "data/jurisiq.faiss"
    duckdb_path: str = "data/jurisiq.duckdb"
    corpus_dir: str = "data/corpus"
    debate_rounds: int = 3

    class Config:
        env_file = ".env"

settings = Settings()