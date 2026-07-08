"""Retrieve: find the chunks most relevant to a question (Phase 2).

Embeds the question with the *same* model used at ingest, then asks Chroma for the
nearest stored chunks. Test it standalone before it's wired into the chatbot:

    python rag.py --persona reviewer "what is the max line length?"
"""

import argparse

from ingest import chroma, embed_texts  # reuse one embedder + one cabinet

TOP_K = 4   # how many chunks to hand the model


def search(collection, query: str, k: int = TOP_K) -> list[dict]:
    """Return the k chunks in `collection` closest in meaning to the query."""
    query_vector = embed_texts([query])[0]
    result = collection.query(query_embeddings=[query_vector], n_results=k)

    # Chroma nests results one level deep (one list per query); we sent one query.
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {"text": doc, "source": meta.get("source", "?"), "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def retrieve(persona: str, query: str, k: int = TOP_K) -> list[dict]:
    """Search a persona's on-disk (baked-in) collection."""
    try:
        collection = chroma.get_collection(persona)
    except Exception:
        raise SystemExit(
            f"No index for '{persona}'. Build it first: python ingest.py --persona {persona}"
        )
    return search(collection, query, k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a persona's document index.")
    parser.add_argument("--persona", required=True, help="Persona index to search")
    parser.add_argument("query", help="The question to search for")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hits = retrieve(args.persona, args.query)

    if not hits:
        print("No chunks found — is the index empty?")
        raise SystemExit(0)

    print(f"Top {len(hits)} chunks for: {args.query!r}\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] source={hit['source']}  distance={hit['distance']:.3f}")
        print(f"    {hit['text'][:200]}...\n")
