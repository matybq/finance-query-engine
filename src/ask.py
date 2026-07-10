"""Command-line interface for corpus-grounded financial Q&A."""

from __future__ import annotations

import argparse

from src.generation.answer import AnswerResult, answer

APP_NAME = "Finance Query Engine"
EXIT_COMMANDS = {"exit", "quit", "q"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finance-ask",
        description="Ask evidence-grounded questions over the indexed financial filing corpus.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. If omitted, an interactive session starts.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved chunks to use as context. Default: 4.",
    )
    return parser


def print_welcome() -> None:
    print(f"\n{APP_NAME}")
    print("Grounded financial Q&A over the indexed corpus.")
    print("\nUsage")
    print('  ./finance-ask "What are Airbnb\'s main risk factors?"')
    print("  ./finance-ask --top-k 6 \"How does Airbnb describe revenue growth?\"")
    print("\nInteractive mode")
    print("  Type a question and press Enter.")
    print("  Type 'exit', 'quit', or 'q' to close the session.")
    print("\nNotes")
    print("  Answers are generated only from retrieved source context.")
    print("  Sources are listed after each answer.\n")


def print_result(result: AnswerResult) -> None:
    print("\nAnswer")
    print(result.answer)

    if result.sources:
        print("\nSources")
        for source, section in result.sources:
            print(f"- {source} ({section})")
    else:
        print("\nSources")
        print("- No sources returned")


def ask_once(question: str, top_k: int) -> None:
    result = answer(question, k=top_k)
    print_result(result)


def run_interactive(top_k: int) -> None:
    print_welcome()

    while True:
        try:
            question = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed.")
            return

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Session closed.")
            return

        ask_once(question, top_k)
        print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be greater than or equal to 1")

    if not args.question:
        run_interactive(args.top_k)
        return

    question = " ".join(args.question).strip()
    ask_once(question, args.top_k)


if __name__ == "__main__":
    main()
