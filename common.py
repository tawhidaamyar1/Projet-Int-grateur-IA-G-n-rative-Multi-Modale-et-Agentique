"""Configuration, modèles d'embedding et accès PGVector."""
import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = os.getenv("PG_DSN", "postgresql://rag:rag@localhost:5432/rag")
TEXT_MODEL = "BAAI/bge-small-en-v1.5"   # embeddings texte open source (384 dim)
CLIP_MODEL = "clip-ViT-B-32"            # CLIP texte+image (512 dim)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")      # génération + tool calling
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:7b")    # description des images
IMG_DIR = "data/images"

_cache = {}


def text_model():
    if "t" not in _cache:
        _cache["t"] = SentenceTransformer(TEXT_MODEL)
    return _cache["t"]


def clip_model():
    if "c" not in _cache:
        _cache["c"] = SentenceTransformer(CLIP_MODEL)
    return _cache["c"]


def conn():
    c = psycopg.connect(DSN, autocommit=True)
    c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(c)
    return c


def init_db():
    with conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            doc TEXT, page INT,
            kind TEXT,              -- 'text' | 'table' | 'image_caption'
            content TEXT,
            meta JSONB DEFAULT '{}',
            embedding vector(384))""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            doc TEXT, page INT, path TEXT, caption TEXT,
            embedding vector(512))""")
