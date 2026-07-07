"""Minimal command-line entry point for asking corpus-grounded questions."""

import sys

from src.generation.answer import answer


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python -m src.ask "Your question"')
        raise SystemExit(1)

    result = answer(" ".join(sys.argv[1:]))
    print(result.answer)
    print("\nSources:")
    for source, section in result.sources:
        print(f"- {source} ({section})")


if __name__ == "__main__":
    main()
