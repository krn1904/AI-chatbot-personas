# Project Plan — Persona Chatbot + RAG ("Chat With Your Documents")

Building **on top of** the existing Blog 2 persona chatbot. Same Python, same
Gemini engine, same phase-by-phase rhythm. We add a knowledge layer (RAG), then a
web UI, then deploy it — all on a $0 path, with AWS kept as an optional extra.

---

## Purpose & scope (read this first)

**Why this exists:** to make the ideas in **Blogs 1–4** concrete by building them.
This is a *learning companion*, not a commercial product. Every decision favours
understanding the RAG concept over scale or polish.

**Deliberately NOT building (yet):** no backend server, no database, no user
accounts/login, no managed cloud services. Those belong to Blog 7 (deployment) and
the eventual capstone — not here. We considered a full FastAPI + Postgres/pgvector
backend and chose against it for this build: it adds infrastructure that teaches
devops, not embeddings/RAG.

**Scope guardrail:** when a choice arises, pick the option that keeps the RAG idea
in the foreground. Simplicity is a feature here, not a limitation.

---

## What we're building (one paragraph)

A persona-driven chatbot that answers from your own documents. You pick a
character (tutor, code reviewer, coach), give it documents, and ask questions in
plain language. Behind the scenes it splits the documents into chunks, turns each
into a meaning-fingerprint (embedding), and stores them in a local search index.
Every question retrieves the most relevant chunks and feeds them to the model,
which replies in the chosen persona's voice and cites which document it used. The
persona controls *how it behaves*; the documents control *what it knows*.

---

## Architecture (the whole system on one page)

```
                        ┌─────────────────────────┐
   documents  ──ingest──▶  Chroma vector index     │   (built once / per upload)
   (PDF, md, txt)        └─────────────────────────┘
                                     ▲
                                     │ nearest chunks
   you ask ──▶ embed question ──────┘
        │                            │
        └──────────────┬─────────────┘
                       ▼
            persona (system msg) + history + retrieved chunks
                       ▼
                  Gemini  ──▶  streamed answer + source citation
```

Two engines, kept separate from the interface so a terminal **or** a web UI can
call them:

- **Ingest engine** — read → chunk → embed → store. Runs ahead of time.
- **Retrieve engine** — embed question → fetch nearest chunks. Runs per question.

---

## Tech stack

| Layer        | Tool                          | Cost  |
|--------------|-------------------------------|-------|
| Language     | Python 3.13                   | free  |
| LLM + embeddings | Google Gemini (`gemini-2.5-flash-lite`, `gemini-embedding-001`) | free tier |
| Vector store | ChromaDB (local, on disk)     | free  |
| PDF parsing  | pypdf                         | free  |
| Config       | python-dotenv                 | free  |
| UI           | Streamlit                     | free  |
| Hosting      | Streamlit Community Cloud (primary) / AWS Lightsail (optional) | $0 / ~$5–10 mo |

*Why Chroma specifically:* it's **embedded** (runs inside Python — no separate
server to run), free, and **swappable** later because retrieval is isolated in
`rag.py`. That fits the keep-it-simple scope. FAISS, pgvector, Qdrant, and Pinecone
only pay off at multi-user/production scale, which is out of scope for this build.

---

## Cost summary

**Primary path = $0.** Gemini free tier + Chroma local + Streamlit Community
Cloud (free, no credit card, deploys from a public GitHub repo).

**AWS option (your $100 credit).** Only if you want hands-on cloud experience for
the portfolio. Use a **fixed-price** service so credit can't run away:

- **AWS Lightsail** container/instance ≈ **$5–10/month** → $100 lasts ~10–20 months.
- Set an **AWS Budgets alert** at $10 and $25 on day one.
- **Avoid** anything that auto-scales or is usage-priced (SageMaker, Bedrock at
  volume, GPU instances) — that's how credits vanish.
- Recommendation: ship free on Streamlit Cloud first; do the AWS deploy later as
  a deliberate learning exercise, not the default.

---

## Phases

Each phase = one concept + one small change + a checkpoint you explain back.
One sitting per phase. (✅ = done)

*Blog map:* Part A is **Blog 4** (embeddings & RAG) made real, resting on the
**Blog 2** engine; the "I don't know" guardrail and treating uploaded docs as
untrusted input also exercise **Blog 3** (prompt safety). Parts B–C go beyond
Blogs 1–4 into UI and deployment — optional extensions, not core learning.

### Part A — RAG core (command line)

**Phase 0 — Setup** ✅
Add `chromadb` + `pypdf`; create `knowledge/<persona>/` folders + sample docs.
*Learn:* why retrieved knowledge is separate from the persona system message.

**Phase 1 — Ingest pipeline** ✅
Read a persona's docs → chunk the text → embed each chunk with Gemini → store in
Chroma. *Learn:* chunking is what quietly decides answer quality. *Files:* new
`ingest.py`. *Decision:* small chunks (~300–500 chars) with ~50-char overlap —
our sample docs are short, fact-dense lines, so small keeps each retrieved piece
precise. *Checkpoint:* explain chunk size vs overlap in your own words.

**Phase 2 — Retrieval** ✅
Embed the question, query Chroma for the nearest chunks. *Learn:* nearest-neighbor
search; why retrieval (not the model) is the weak link. *Files:* new `rag.py`.
*Checkpoint:* why we over-fetch then trim.

**Phase 3 — Wire RAG into the chat loop** ✅
Inject retrieved chunks into the prompt before the existing `generate_content_stream`
call. Loop, memory, streaming unchanged. *Files:* `chatbot.py` (~15 lines).
*Checkpoint:* trace one question end to end.

**Phase 4 — Per-persona knowledge bases** ✅
Each persona gets its own Chroma collection, so tutor/reviewer/coach retrieve from
their own docs. *Learn:* persona = behavior dial, RAG = knowledge dial, independent.
*Files:* `ingest.py`, `rag.py`. *Checkpoint:* what happens if a persona has no docs.

**Phase 5 — Sources & polish** ✅
Show which document each answer came from; handle "no relevant match" / "I don't
know"; tidy errors. *Files:* `chatbot.py`, `rag.py`. *Checkpoint:* make it admit
when the docs don't cover a question.

### Part B — User interface

**Phase 6 — Streamlit UI** ✅
A web page: pick a persona, chat in a proper chat window, see streamed answers and
source citations. Calls the *same* ingest/retrieve engines — no logic rewritten.
*Files:* new `app.py`. *Checkpoint:* the CLI and UI share one engine.

**Phase 7 — Upload your own documents** ✅
Add an upload box: users bring their own files, indexed **in-memory for the
session** (private, no persistence needed, free). Curated personas still ship with
their baked-in docs. **Precedence: user uploads win.** Keep uploads in a separate
session-scoped collection; on each question search the user's collection first and
fall back to the persona's baked-in docs only when the upload has no relevant match.
Tag every retrieved chunk with its source and instruct the model to prefer
user-uploaded chunks whenever they conflict with the defaults. *Files:* `app.py`,
`ingest.py`, `rag.py`. *Checkpoint:* why session-only uploads dodge the
persistence/cost problem; show that an uploaded rule overrides a conflicting
baked-in rule.
*Safety (Blog 3):* uploaded files are untrusted input — fence retrieved chunks as
data, not instructions, so a malicious document can't hijack the persona (prompt
injection). This matters more now that uploads win precedence.

### Part C — Deploy

**Phase 8 — Make it deploy-ready** ✅
Configurable paths/keys via env, rebuild-index-on-boot for curated docs, lock
`requirements.txt`, `.gitignore` hygiene, README run instructions. *Checkpoint:*
never depend on the server's disk surviving a restart.

**Phase 9 — Deploy (free first)**
Push to a public GitHub repo → deploy on Streamlit Community Cloud (free). Optional
follow-up: AWS Lightsail with a Budgets alarm, as a cloud-experience exercise.
*Checkpoint:* a stranger can open the URL and use it.

### Part D — Write-up

**Phase 10 — Blog handover**
Fold the build into Blog 4's pending code-walkthrough (same format as the Blog 2
handover): phase-by-phase, finished version in pieces, with a companion build log.

---

## How users will use it

- **Now (CLI):** drop docs in a persona folder → run `python ingest.py` once →
  `python chatbot.py --persona reviewer` → ask questions, see sources.
- **Later (web UI):** open the URL → pick a persona (or upload your own files) →
  chat → get answers grounded in those documents with citations. No setup.

---

## Time estimate (building together, to learn)

~8–12 focused hours total across phases: ~6–9h for the RAG core (A), ~2–3h for the
UI (B), ~1–2h for deploy (C). One phase per sitting.
