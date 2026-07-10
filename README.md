# Persona Chatbot + RAG — Chat With Your Documents

A persona-driven chatbot that answers from **your own documents**. Pick a character
(tutor, code reviewer, coach), give it documents, and ask questions in plain
language — it retrieves the relevant passages and answers from them, citing the
source. Runs as a command-line tool **and** a web app, on Google Gemini's free tier.

Built phase-by-phase: it started as the Blog 2 persona chatbot and grew into the
Blog 4 "chat with your documents" (RAG) project. Behaviour lives in the persona;
knowledge lives in the documents.

## Features

- **Personas** — swap the character with a text file (`--persona reviewer`).
- **Chat with documents (RAG)** — chunk → embed → retrieve → answer, per persona.
- **Bring your own files** — upload PDFs/text in the web app (session-only, private).
- **Uploads win** — your documents override a persona's baked-in ones.
- **Honest citations** — sources shown only when the answer really used the docs.
- **Guardrails** — falls back to "I don't know" / general knowledge when off-topic;
  uploaded files are treated as data, not instructions (prompt-injection safe).
- **One engine, two front-ends** — `chatbot.py` (terminal) and `app.py` (web) both
  call `engine.py`.

## Quick start

```bash
git clone https://github.com/krn1904/AI-chatbot-personas.git
cd AI-chatbot-personas

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Add a free [Google AI Studio](https://aistudio.google.com/apikey) key:

```bash
cp .env.example .env               # then paste your key into .env
```

Build the document indexes (once, or whenever docs change):

```bash
python ingest.py --all
```

Run it:

```bash
python chatbot.py --persona reviewer    # command line
streamlit run app.py                     # web app (http://localhost:8501)
```

## Using it

**Command line:** pick a persona, ask questions, see a `Sources:` line when an
answer came from the docs. `python chatbot.py --list` shows the personas.

**Web app:** choose a persona from the dropdown, optionally upload your own files in
the sidebar, and chat. Each persona keeps its own conversation.

## Give a persona documents

Drop `.md`, `.txt`, or `.pdf` files into `knowledge/<persona>/` and re-run
`python ingest.py --all`. For example, `knowledge/reviewer/` holds the code
reviewer's style guide.

## Deploy free (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repo
   with `app.py` as the entry point.
3. In **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
4. Deploy. The baked-in indexes rebuild automatically on startup from the committed
   `knowledge/` folders — nothing depends on the server's disk surviving a restart.

## How it works (RAG in one paragraph)

The model is never retrained on your data. Ahead of time, each document is split
into chunks and turned into a "meaning fingerprint" (embedding) stored in a local
vector index (Chroma). At question time the question is embedded, the nearest chunks
are retrieved, and those chunks are pasted into the prompt — so the model answers
from material it never trained on. Uploaded files are searched first; the persona's
baked-in docs are the fallback.

## Project structure

```
├── engine.py            # shared core: personas, retrieval, docs-first prompting
├── chatbot.py           # command-line front-end
├── app.py               # Streamlit web front-end
├── ingest.py            # build indexes (on-disk baked-in + in-memory uploads)
├── rag.py               # search a vector collection
├── personas/            # one .txt = one personality
├── knowledge/<persona>/ # documents a persona can answer from
├── requirements.txt
├── .env.example         # local API key template
├── .streamlit/          # secrets.toml.example for deploy
└── PROJECT_PLAN.md      # phase-by-phase build plan
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `models/... not found` for embeddings | Embedding model retired — set `EMBED_MODEL` in `ingest.py` to a current one (e.g. `gemini-embedding-001`) |
| `429` with `limit: 0` | Use `gemini-2.5-flash` or `gemini-2.5-flash-lite` |
| `429`, retry in 60s | Temporary rate limit — wait a minute |
| `No index for '<persona>'` | Run `python ingest.py --all` |
| `Set GEMINI_API_KEY` | Copy `.env.example` to `.env` and add your key |

## License

MIT — use freely for learning and blog follow-alongs.
