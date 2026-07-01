"""Persona chatbot — the terminal front-end over the shared engine (engine.py)."""

import argparse

from google.genai import errors, types

from engine import (
    DEFAULT_PERSONA,
    MODEL,
    AnswerStream,
    build_config,
    get_relevant_context,
    list_personas,
    load_persona,
    with_context,
)

QUIT_WORDS = {"quit", "exit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persona chatbot — swap the system message, get a new tool."
    )
    parser.add_argument(
        "--persona",
        default=DEFAULT_PERSONA,
        help=f"Persona file name from personas/ (default: {DEFAULT_PERSONA})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available personas and exit",
    )
    return parser.parse_args()


def stream_to_terminal(stream: AnswerStream) -> None:
    """Print the streamed answer, printing the 'Bot: ' prefix only once text arrives."""
    started = False
    for piece in stream:
        if not piece:
            continue
        if not started:
            print("Bot: ", end="", flush=True)
            started = True
        print(piece, end="", flush=True)
    if started:
        print("\n")


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        print("Available personas:")
        for name in list_personas():
            print(f"  - {name}")
        raise SystemExit(0)

    system_prompt = load_persona(args.persona)
    generation_config = build_config(system_prompt)

    print(f"Chat started (persona: {args.persona}). Type 'quit' or 'exit' to leave.\n")

    messages: list[types.Content] = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in QUIT_WORDS:
            print("Bye!")
            break

        messages.append(types.UserContent(user_input))

        # RAG: pull relevant chunks (gated by distance) and attach to this turn only.
        context, sources = get_relevant_context(args.persona, user_input)
        call_contents = messages[:-1] + [with_context(context, user_input)]

        stream = AnswerStream(call_contents, generation_config)
        try:
            stream_to_terminal(stream)
        except errors.ClientError as e:
            messages.pop()
            if e.code == 429:
                print(
                    "Bot: Rate limit hit. Wait ~60 seconds and try again.\n"
                    f"     (Free tier: {MODEL} has daily/minute caps.)\n"
                )
            else:
                print(f"Bot: API error ({e.code}). {e.message}\n")
            continue
        except errors.ServerError as e:
            messages.pop()
            print(f"Bot: Server busy ({e.code}). Try again in a moment.\n")
            continue

        if stream.used_docs and sources:
            print(f"Sources: {', '.join(sources)}\n")

        messages.append(types.ModelContent(stream.text))
