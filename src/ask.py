"""Command-line interface for corpus-grounded Q&A."""

from __future__ import annotations

import argparse
import sys

from src.agent.graph import AgentResult, run

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
        "--debug",
        action="store_true",
        help="Print routing and retrieval trace metadata after the answer.",
    )
    return parser


def print_welcome() -> None:
    print(f"\n{APP_NAME}")
    print("Grounded financial Q&A over the indexed corpus.")
    print("\nUsage")
    print('  uv run finance-ask "What are Airbnb\'s main risk factors?"')
    print('  uv run finance-ask --debug "How does Airbnb make money?"')
    print("\nInteractive mode")
    print("  Type a question and press Enter.")
    print("  Type 'exit', 'quit', or 'q' to close the session.")
    print("\nNotes")
    print("  Answers are generated only from retrieved source context.")
    print("  Sources are listed after each answer when evidence is used.\n")


def print_debug(result: AgentResult) -> None:
    print("\nDebug")
    print(f"- route: {result.route}")
    if result.route_reason:
        print(f"- route_reason: {result.route_reason}")

    if result.rewritten_queries:
        print("- rewritten_queries:")
        for query in result.rewritten_queries:
            print(f"  - {query}")

    if result.retrieval_attempts:
        print("- retrieval_attempts:")
        for query, chunk_ids in result.retrieval_attempts:
            chunks = ", ".join(chunk_ids) if chunk_ids else "no chunks"
            print(f"  - {query} -> {chunks}")

    if result.grading_attempts:
        print("- grading_attempts:")
        for query, sufficient, reason in result.grading_attempts:
            print(f"  - {query} -> sufficient={sufficient}: {reason}")


def print_result(result: AgentResult, debug: bool = False) -> None:
    print("\nAnswer")
    print(result.answer)

    if result.sources:
        print("\nSources")
        for source, section in result.sources:
            print(f"- {source} ({section})")

    if debug:
        print_debug(result)


def ask_once(question: str, debug: bool = False) -> None:
    result = run(question)
    print_result(result, debug=debug)


def run_interactive(debug: bool = False) -> None:
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

        try:
            ask_once(question, debug=debug)
        except Exception as error:
            print(
                f"Error: the agent failed to process the question ({error.__class__.__name__}). Try again.",
                file=sys.stderr,
            )
        print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.question:
        run_interactive(debug=args.debug)
        return

    question = " ".join(args.question).strip()
    if not question:
        parser.error("question must not be empty")

    try:
        ask_once(question, debug=args.debug)
    except Exception as error:
        print(
            f"Error: the agent failed to process the question ({error.__class__.__name__}). Try again.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
