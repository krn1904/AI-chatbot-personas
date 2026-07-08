"""Ingest: turn a persona's documents into a searchable Chroma index (Phase 1).

Reads knowledge/<persona>/*, splits each file into chunks, embeds the chunks with
Gemini, and stores them in a Chroma collection named after the persona. Run once
per persona, or whenever its documents change:

    python ingest.py --persona reviewer
    python ingest.py --list
"""

import argparse
import io
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_key_here":
    raise SystemExit(
        "Set GEMINI_API_KEY in .env — get one free at https://aistudio.google.com/apikey"
    )

client = genai.Client(api_key=api_key)

# Tunable knobs.
EMBED_MODEL = "gemini-embedding-001"   # Gemini embeddings, free tier
CHUNK_SIZE = 500                     # characters per chunk
CHUNK_OVERLAP = 50                   # characters repeated between neighbours
EMBED_BATCH = 100                    # chunks sent per embedding request

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
CHROMA_DIR = Path(__file__).parent / "chroma_db"   # the filing cabinet, on disk
READABLE_SUFFIXES = {".txt", ".md", ".pdf"}

chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))


def list_personas() -> list[str]:
    if not KNOWLEDGE_DIR.is_dir():
        return []
    return sorted(p.name for p in KNOWLEDGE_DIR.iterdir() if p.is_dir())


def personas_with_docs() -> list[str]:
    """Persona folders that actually contain at least one readable document."""
    return [
        name for name in list_personas()
        if any(p.suffix.lower() in READABLE_SUFFIXES
               for p in (KNOWLEDGE_DIR / name).iterdir())
    ]


def read_file(path: Path) -> str:
    """Return the plain text of one document, or '' if it can't be read."""
    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:  # a corrupt PDF shouldn't kill the whole ingest
            print(f"  ! skipped {path.name}: {e}")
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_upload(name: str, data: bytes) -> str:
    """Extract text from an uploaded file's raw bytes (PDF, txt, or md)."""
    if name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"  ! skipped {name}: {e}")
            return ""
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping fixed-size character windows."""
    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with Gemini, batched so we stay friendly to rate limits."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        result = client.models.embed_content(model=EMBED_MODEL, contents=batch)
        vectors.extend(e.values for e in result.embeddings)
    return vectors


def build_memory_index(files: list[tuple[str, bytes]]):
    """Chunk + embed uploaded files into a fresh in-memory Chroma collection.

    Returns the collection, or None if nothing readable was found. Uses an
    EphemeralClient, so the index lives only in memory and vanishes with the session.
    """
    chunks: list[str] = []
    sources: list[str] = []
    for name, data in files:
        for chunk in chunk_text(read_upload(name, data)):
            chunks.append(chunk)
            sources.append(name)

    if not chunks:
        return None

    embeddings = embed_texts(chunks)
    memory = chromadb.EphemeralClient()
    collection = memory.create_collection("uploads")
    collection.add(
        ids=[f"upload-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": s} for s in sources],
    )
    return collection


def ingest_persona(persona: str) -> None:
    folder = KNOWLEDGE_DIR / persona
    if not folder.is_dir():
        raise SystemExit(f"No knowledge folder for '{persona}'. Expected {folder}")

    docs = [p for p in sorted(folder.iterdir())
            if p.suffix.lower() in READABLE_SUFFIXES]
    if not docs:
        raise SystemExit(f"No documents in {folder} (add .txt, .md, or .pdf files).")

    # Keep each chunk paired with its source filename, for citation later.
    chunks: list[str] = []
    sources: list[str] = []
    for doc in docs:
        for chunk in chunk_text(read_file(doc)):
            chunks.append(chunk)
            sources.append(doc.name)

    if not chunks:
        raise SystemExit(f"Read 0 chunks from {folder} — are the files empty?")

    print(f"Embedding {len(chunks)} chunks from {len(docs)} file(s)...")
    embeddings = embed_texts(chunks)

    # Rebuild the collection from scratch so re-running never duplicates chunks.
    try:
        chroma.delete_collection(persona)
    except Exception:
        pass  # nothing to delete on first run
    collection = chroma.create_collection(persona)
    collection.add(
        ids=[f"{persona}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": s} for s in sources],
    )
    print(f"Done. '{persona}' now holds {collection.count()} chunks in {CHROMA_DIR.name}/.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a persona's document index.")
    parser.add_argument("--persona", help="Persona folder under knowledge/ to ingest")
    parser.add_argument("--list", action="store_true",
                        help="List personas that have a knowledge folder")
    parser.add_argument("--all", action="store_true",
                        help="Ingest every persona that has documents")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        print("Personas with knowledge folders:")
        for name in list_personas():
            print(f"  - {name}")
        raise SystemExit(0)

    if args.all:
        personas = personas_with_docs()
        if not personas:
            raise SystemExit("No personas have documents under knowledge/.")
        for name in personas:
            print(f"\n=== {name} ===")
            ingest_persona(name)
        raise SystemExit(0)

    if not args.persona:
        raise SystemExit(
            "Pass --persona <name> (or --list). e.g. python ingest.py --persona reviewer"
        )

    ingest_persona(args.persona)
