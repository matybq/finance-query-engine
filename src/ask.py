"""Minimal command-line interface for corpus-grounded Q&A."""

from __future__ import annotations

import sys

from src.agent.graph import run

USAGE = 'Usage: uv run python -m src.ask "question"'


def main() -> None:
    if len(sys.argv) == 1:
        print(USAGE)
        sys.exit(1)

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print(USAGE)
        sys.exit(1)

    try:
        result = run(question)
    except Exception as error:
        print(
            f"Error: the agent failed to process the question ({error.__class__.__name__}). Try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(result.answer)

    if result.sources:
        print("\nSources:")
        for source, section in result.sources:
            print(f"- {source} ({section})")


if __name__ == "__main__":
    main()
