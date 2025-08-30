"""Aggregate judged generations into detailed results and a summary.

Usage:
    python result.py generations.judged.jsonl -o out.jsonl [--data data/vibe-eval.v1.jsonl]

This script expects the generations file to already contain judgement fields
from an external judge, specifically "score" (0/1) and optional "explanation".
It will:
  - Join with the dataset by example_id to attach category/prompt/reference/image
  - Write a detailed results JSONL to --output
  - Print a README-style table (percent scores, 0-100)
  - Write a summary JSON with per-category and overall percentages
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate judged generations without external API calls.")
    parser.add_argument(
        "generations",
        type=Path,
        help="Path to judged generations JSONL with keys example_id, generation, score, explanation.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data/vibe-eval.v1.jsonl",
        help="Path to dataset JSONL (default: data/vibe-eval.v1.jsonl).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path for detailed results JSONL.",
    )
    parser.add_argument(
        "--output_summary",
        type=Path,
        default=None,
        help="Optional output path for summary JSON (defaults to --output with _summary suffix).",
    )
    return parser.parse_args()


def build_dataset_index(dataset_path: Path) -> Dict[str, dict]:
    dataset_index: Dict[str, dict] = {}
    for example in load_jsonl(dataset_path):
        example_id = example.get("example_id") or example.get("id") or example.get("exampleId")
        if not example_id:
            continue
        dataset_index[example_id] = example
    return dataset_index


def compute_percentages(rows: List[dict]) -> Tuple[Dict[str, float], float]:
    scores_by_category: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        category = row.get("category") or row.get("category_name") or "unknown"
        score = row.get("score")
        if score is None:
            continue
        try:
            numeric = float(score)
        except Exception:
            continue
        scores_by_category[category].append(numeric)

    percentages: Dict[str, float] = {}
    for category, scores in scores_by_category.items():
        if not scores:
            continue
        percentages[category] = 100.0 * (sum(scores) / len(scores))

    # overall
    all_scores = [s for lst in scores_by_category.values() for s in lst]
    overall = 100.0 * (sum(all_scores) / len(all_scores)) if all_scores else 0.0
    return percentages, overall


def main() -> None:
    args = parse_args()

    dataset_index = build_dataset_index(args.data)

    detailed_rows: List[dict] = []
    missing = 0
    for row in load_jsonl(args.generations):
        example_id = row.get("example_id") or row.get("id") or row.get("exampleId")
        if not example_id:
            continue
        ds = dataset_index.get(example_id)
        if ds is None:
            missing += 1
            continue
        detailed_rows.append(
            {
                "example_id": example_id,
                "category": ds.get("category"),
                "prompt": ds.get("prompt"),
                "reference": ds.get("reference") or ds.get("rubric"),
                "image": ds.get("media_url") or ds.get("image"),
                "generation": row.get("generation") or row.get("response"),
                "score": row.get("score"),
                "explanation": row.get("explanation") or row.get("evaluator_explanation"),
            }
        )

    if missing:
        print(f"Warning: {missing} judged rows had example_ids not found in dataset {args.data}")

    # Write detailed results
    save_jsonl(args.output, detailed_rows)
    print(f"Output {len(detailed_rows)} examples to {args.output}.")

    # Compute and print README-style table
    per_category, overall = compute_percentages(detailed_rows)
    print("\n| Category           |  Score (%)   |")
    print("|--------------------|--------------|")
    for category in sorted(per_category.keys()):
        print(f"| {category.ljust(18)} | {per_category[category]:6.2f}       |")
    print(f"| ALL                | {overall:6.2f}       |\n")

    # Write summary JSON
    out_summary: Path
    if args.output_summary is not None:
        out_summary = args.output_summary
    else:
        base, ext = os.path.splitext(str(args.output))
        out_summary = Path(base + "_summary" + ext)
    summary_payload = {**per_category, "overall": overall}
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w") as f:
        json.dump(summary_payload, f)
    print(f"Wrote summary to {out_summary}.")


if __name__ == "__main__":
    main()


