"""Runbook search: Chroma vector database (local, persistent)."""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

ROOT = Path(__file__).resolve().parent.parent
_ef = ONNXMiniLM_L6_V2()
_ef.DOWNLOAD_PATH = ROOT / ".data/models/all-MiniLM-L6-v2"   # keep the model inside the project


def collection():
    client = chromadb.PersistentClient(path=str(ROOT / ".data/chroma"))
    col = client.get_or_create_collection("runbooks", embedding_function=_ef)
    if col.count() == 0:
        docs = sorted((ROOT / "runbooks").glob("*.md"))
        col.add(ids=[d.stem.split("-", 2)[0] + "-" + d.stem.split("-", 2)[1] for d in docs],
                documents=[d.read_text() for d in docs], metadatas=[{"file": d.name} for d in docs])
    return col


def search(query: str, k: int = 2) -> list[dict]:
    r = collection().query(query_texts=[query], n_results=k)
    return [{"id": i, "text": d[:600], "distance": round(float(s), 3)} for i, d, s in zip(r["ids"][0], r["documents"][0], r["distances"][0])]


if __name__ == "__main__":
    for q in ("several access routers lose packets at the same time", "bgp neighbor keeps going down and up", "cpu spike from backup job on one router"):
        print(q, "→", [(h["id"], h["distance"]) for h in search(q)])
