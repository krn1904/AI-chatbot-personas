# AI Chatbot Personas

A small command-line chatbot with swappable personalities, built phase-by-phase as the hands-on project for **Blog 2: Calling an LLM API**.

Same engine — different system message — different product. No server, no database, ~140 lines of Python.

## Features

- **Google Gemini** on the free tier (`gemini-2.5-flash-lite`)
- **Conversation memory** — you hold history and resend it each turn
- **Streaming replies** — text appears live in the terminal
- **Persona swapping** — tutor, interview coach, recipe helper, code reviewer
- **Error handling** — friendly messages on rate limits and API errors

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/krn1904/AI-chatbot-personas.git
cd AI-chatbot-personas

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) (no credit card required).

```bash
cp .env.example .env
```

Open `.env` and set:

```
GEMINI_API_KEY=your_actual_key_here
```

Never commit `.env` — it is already in `.gitignore`.

### 3. Run

```bash
python chatbot.py
```

Type `quit` or `exit` to leave.

## Personas

| Persona | Command | What it does |
|---------|---------|----------------|
| Tutor (default) | `python chatbot.py` | Friendly Blog 2 concepts tutor |
| Interview coach | `python chatbot.py --persona coach` | Mock interview feedback |
| Recipe helper | `python chatbot.py --persona recipe` | Suggests recipes from ingredients |
| Code reviewer | `python chatbot.py --persona reviewer` | Terse, direct code feedback |

List all personas:

```bash
python chatbot.py --list
```

### Add your own

1. Create `personas/mybot.txt` with a system prompt.
2. Run `python chatbot.py --persona mybot`.

No code changes required.

## Project structure

```
├── chatbot.py           # main app
├── personas/            # one .txt file = one personality
├── requirements.txt
├── .env.example         # template for your API key
├── README.md            # you are here
└── BLOG_HANDOVER.md     # full phase-by-phase build guide
```

## How it maps to Blog 2

| Concept | In this repo |
|---------|----------------|
| Config & auth | `.env` + `genai.Client()` |
| Chat completion | `generate_content_stream()` |
| Memory | `messages` list |
| Persona | `personas/*.txt` → `system_instruction` |
| Streaming | `stream_reply()` |
| Control | `temperature`, `max_output_tokens` |
| Error handling | `try/except` on API calls |

## Build walkthrough

This repo was built in six phases (setup → hello world → loop → memory → persona → polish → swap personas). The full learning handover — concepts, checkpoints, gotchas, and blog notes — is in **[BLOG_HANDOVER.md](BLOG_HANDOVER.md)**.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `429` with `limit: 0` | Switch model to `gemini-2.5-flash` or `gemini-2.5-flash-lite` in `chatbot.py` |
| `429` with “retry in 60s” | Temporary rate limit — wait a minute and try again |
| `Set GEMINI_API_KEY in .env` | Copy `.env.example` to `.env` and add your key |

Free-tier limits: [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)

## Requirements

- Python 3.9+
- `google-genai`
- `python-dotenv`

## License

MIT — use freely for learning and blog follow-alongs.
