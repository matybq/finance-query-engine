"""Run a report-only RAGAS evaluation against the LangGraph agent."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.graph import AgentResult, run
from src.config import get_settings
from src.generation.llm import build_llm

CASES_PATH = Path(__file__).with_name("ragas_cases.jsonl")
EXPERIMENTS_DIR = Path(__file__).with_name("experiments")
METRIC_COLUMNS = ("faithfulness", "context_recall", "factual_correctness(mode=f1)")
METRIC_LABELS = {"factual_correctness(mode=f1)": "factual_correctness"}


def import_ragas():
    """Import RAGAS lazily so missing eval dependencies get a helpful message."""
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper

        warnings.filterwarnings(
            "ignore",
            message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"LangchainLLMWrapper is deprecated.*",
            category=DeprecationWarning,
        )
        from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "RAGAS evaluation dependencies are missing. Install them with: "
            "uv sync --extra eval"
        ) from exc

    return {
        "EvaluationDataset": EvaluationDataset,
        "FactualCorrectness": FactualCorrectness,
        "Faithfulness": Faithfulness,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "LLMContextRecall": LLMContextRecall,
        "evaluate": evaluate,
    }


def load_cases(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            case = json.loads(line)
            missing_fields = {"id", "question", "reference"} - set(case)
            if missing_fields:
                raise ValueError(
                    f"{path}:{line_number} missing required fields: {sorted(missing_fields)}"
                )

            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break

    return cases


def source_label(source: tuple[str, str]) -> str:
    source_name, section = source
    return f"{source_name} ({section})"


def collect_sample(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    result: AgentResult = run(case["question"])
    sample = {
        "user_input": case["question"],
        "response": result.answer,
        "retrieved_contexts": result.retrieved_contexts,
        "reference": case["reference"],
    }
    metadata = {
        "id": case["id"],
        "family": case.get("family", "uncategorized"),
        "route": result.route,
        "sources": "; ".join(source_label(source) for source in result.sources),
        "retrieved_chunk_ids": ", ".join(result.retrieved_chunk_ids),
        "rewritten_queries": " | ".join(result.rewritten_queries),
    }
    return sample, metadata


def collect_samples(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = []
    metadata = []

    for case in cases:
        print(f"Collecting sample: {case['id']}")
        sample, sample_metadata = collect_sample(case)
        samples.append(sample)
        metadata.append(sample_metadata)

    return samples, metadata


def build_metrics(ragas_modules: dict[str, Any]) -> list[Any]:
    return [
        ragas_modules["Faithfulness"](),
        ragas_modules["LLMContextRecall"](),
        ragas_modules["FactualCorrectness"](),
    ]


def evaluate_samples(samples: list[dict[str, Any]], ragas_modules: dict[str, Any]):
    settings = get_settings()
    evaluator_llm = ragas_modules["LangchainLLMWrapper"](build_llm(settings))
    evaluation_dataset = ragas_modules["EvaluationDataset"].from_list(samples)

    return ragas_modules["evaluate"](
        dataset=evaluation_dataset,
        metrics=build_metrics(ragas_modules),
        llm=evaluator_llm,
        raise_exceptions=False,
    )


def attach_metadata(dataframe, metadata: list[dict[str, Any]]):
    for column in ("rewritten_queries", "retrieved_chunk_ids", "sources", "route", "family", "id"):
        dataframe.insert(0, column, [item[column] for item in metadata])
    return dataframe


def print_summary(dataframe) -> None:
    print("\nRAGAS summary")
    for metric in METRIC_COLUMNS:
        if metric in dataframe:
            label = METRIC_LABELS.get(metric, metric)
            print(f"{label:<22} {dataframe[metric].mean():.3f}")

    columns = ["id", *[metric for metric in METRIC_COLUMNS if metric in dataframe]]
    if columns:
        print("\nPer-case scores")
        print(dataframe[columns].to_string(index=False))


def save_results(dataframe, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"ragas_{timestamp}.csv"
    dataframe.to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run report-only RAGAS metrics over the finance query agent."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASES_PATH,
        help=f"JSONL cases file. Defaults to {CASES_PATH}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases to control cost while iterating.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_DIR,
        help=f"Directory for CSV reports. Defaults to {EXPERIMENTS_DIR}.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print scores without writing a CSV report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")

    ragas_modules = import_ragas()
    cases = load_cases(args.cases, limit=args.limit)
    if not cases:
        raise SystemExit(f"No RAGAS cases found in {args.cases}")

    samples, metadata = collect_samples(cases)
    result = evaluate_samples(samples, ragas_modules)
    dataframe = attach_metadata(result.to_pandas(), metadata)

    print_summary(dataframe)
    if not args.no_save:
        output_path = save_results(dataframe, args.output_dir)
        print(f"\nSaved CSV report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
