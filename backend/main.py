import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

app = FastAPI(title="RepoSage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    client = chromadb.Client()
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    col = client.create_collection(name="health_check", embedding_function=embed_fn)

    col.upsert(
        ids=["test_chunk"],
        documents=["def hello(): return 'world'"],
        metadatas=[{"file": "test.py", "lines": "1-1", "language": "py"}],
    )

    results = col.query(query_texts=["hello function"], n_results=1)
    retrieved = len(results["documents"][0])

    client.delete_collection("health_check")

    return {
        "status": "ok",
        "rag_pipeline": "operational",
        "retrieved_chunks": retrieved,
    }
