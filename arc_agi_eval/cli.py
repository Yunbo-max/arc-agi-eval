from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .baseline import default_result_paths, run_baseline
from .dataset import DatasetError, enumerate_dataset, group_by_split, task_files
from .firewall import FirewallError, generate_challenge_tree
from .scoring import PredictionValidationError, score_prediction_file
from .validation import TaskValidationError, load_task


def _validate_command(args: argparse.Namespace) -> int:
    files: list[Path] = []
    seen: set[Path] = set()
    for supplied_path in args.paths:
        for path in task_files(supplied_path, recursive=Path(supplied_path).is_dir()):
            normalized = path.resolve()
            if normalized not in seen:
                seen.add(normalized)
                files.append(path)

    errors: list[str] = []
    for path in sorted(files):
        try:
            load_task(path, require_test_outputs=not args.allow_missing_test_outputs)
        except TaskValidationError as exc:
            errors.append(str(exc))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(
            f"validation failed: {len(errors)} of {len(files)} task(s) invalid",
            file=sys.stderr,
        )
        return 1
    print(f"validated {len(files)} task(s)")
    return 0


def _list_command(args: argparse.Namespace) -> int:
    refs = enumerate_dataset(args.dataset)
    grouped = group_by_split(refs)
    if args.json:
        payload = {
            split: [ref.task_id for ref in split_refs]
            for split, split_refs in grouped.items()
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.tasks:
        for ref in refs:
            print(f"{ref.split}\t{ref.task_id}\t{ref.path}")
    else:
        for split, split_refs in grouped.items():
            print(f"{split}\t{len(split_refs)}")
        print(f"total\t{len(refs)}")
    return 0


def _score_command(args: argparse.Namespace) -> int:
    score = score_prediction_file(args.predictions, args.task_dir, top_k=args.top_k)
    if args.json:
        print(json.dumps(score.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"top_k: {score.top_k}")
        print(
            f"primary_output_exact_pass@{score.top_k}: "
            f"{score.outputs_exact}/{score.outputs_total} "
            f"({score.output_exact_accuracy:.6f})"
        )
        print(
            f"secondary_strict_task_exact@{score.top_k}: "
            f"{score.tasks_exact}/{score.tasks_total} "
            f"({score.task_exact_accuracy:.6f})"
        )
        print(
            f"diagnostic_cell_accuracy: {score.cells_correct}/{score.cells_total} "
            f"({score.cell_accuracy:.6f})"
        )
        print(f"tasks_predicted: {score.tasks_predicted}/{score.tasks_total}")
    return 0


def _baseline_command(args: argparse.Namespace) -> int:
    default_predictions, default_metadata = default_result_paths(args.task_dir)
    prediction_path = Path(args.output) if args.output else default_predictions
    metadata_path = Path(args.metadata) if args.metadata else default_metadata
    metadata = run_baseline(
        args.task_dir,
        prediction_path,
        metadata_path,
        score=args.score,
        top_k=2,
    )
    if args.json:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    else:
        print(f"predictions: {metadata['prediction_path']}")
        print(f"metadata: {metadata_path.resolve()}")
        print(f"tasks: {metadata['tasks_total']}")
        print(f"test_outputs: {metadata['test_outputs_total']}")
        print(f"wall_time_seconds: {metadata['wall_time_seconds']:.6f}")
        if metadata["score"] is not None:
            result = metadata["score"]
            print(
                f"primary_output_exact_pass@{result['top_k']}: "
                f"{result['outputs_exact']}/{result['outputs_total']} "
                f"({result['output_exact_accuracy']:.6f})"
            )
            print(
                f"secondary_strict_task_exact@{result['top_k']}: "
                f"{result['tasks_exact']}/{result['tasks_total']} "
                f"({result['task_exact_accuracy']:.6f})"
            )
    return 0


def _challenge_command(args: argparse.Namespace) -> int:
    manifest = generate_challenge_tree(args.source, args.destination)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"generated {manifest['tasks_total']} challenge-only task(s)")
        print(f"destination: {Path(args.destination).resolve()}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arc-agi-eval",
        description="Validate, enumerate, and score ARC-AGI benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate ARC task JSON")
    validate_parser.add_argument("paths", nargs="+", help="task files or directories")
    validate_parser.add_argument(
        "--allow-missing-test-outputs",
        action="store_true",
        help="allow unlabeled test pairs that omit output",
    )
    validate_parser.set_defaults(handler=_validate_command)

    list_parser = subparsers.add_parser("list", help="enumerate dataset splits and tasks")
    list_parser.add_argument("dataset", help="dataset root or split directory")
    list_output = list_parser.add_mutually_exclusive_group()
    list_output.add_argument(
        "--tasks", action="store_true", help="print every task ID and path"
    )
    list_output.add_argument("--json", action="store_true", help="emit task IDs as JSON")
    list_parser.set_defaults(handler=_list_command)

    score_parser = subparsers.add_parser("score", help="score an ARC prediction file")
    score_parser.add_argument("predictions", help="prediction JSON file")
    score_parser.add_argument("task_dir", help="labeled split directory")
    score_parser.add_argument(
        "--top-k", type=int, default=2, help="attempt budget (default: 2)"
    )
    score_parser.add_argument("--json", action="store_true", help="emit metrics as JSON")
    score_parser.set_defaults(handler=_score_command)

    baseline_parser = subparsers.add_parser(
        "baseline", help="generate deterministic Top-2 baseline predictions"
    )
    baseline_parser.add_argument("task_dir", help="ARC split directory")
    baseline_parser.add_argument(
        "--output", help="prediction JSON path (default: results directory)"
    )
    baseline_parser.add_argument(
        "--metadata", help="run metadata JSON path (default: results directory)"
    )
    baseline_parser.add_argument(
        "--score",
        action="store_true",
        help="post-hoc score against labels after writing predictions",
    )
    baseline_parser.add_argument(
        "--json", action="store_true", help="emit run metadata as JSON"
    )
    baseline_parser.set_defaults(handler=_baseline_command)

    challenge_parser = subparsers.add_parser(
        "challenge", help="generate a test-label-free ARC task tree"
    )
    challenge_parser.add_argument("source", help="labeled split or dataset")
    challenge_parser.add_argument("destination", help="new or empty output directory")
    challenge_parser.add_argument("--json", action="store_true", help="emit manifest JSON")
    challenge_parser.set_defaults(handler=_challenge_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (
        DatasetError,
        FirewallError,
        PredictionValidationError,
        TaskValidationError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
