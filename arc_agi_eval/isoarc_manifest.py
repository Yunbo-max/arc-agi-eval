"""Build the frozen 64-base-task input-visible IsoARC design."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .isoarc import (
    D4,
    INVERSE_D4,
    color_transform,
    inverse_color_mapping,
    restore_test_order,
    transform_predictions,
    transform_task,
)
from .firewall import challenge_only
from .scoring import score_predictions
from .validation import Grid, load_task


OFFICIAL_PROFILE_ID = "arc-rebench-isoarc-fixed64-v1"
OFFICIAL_QUOTAS = {"arc_agi_1": 32, "arc_agi_2": 32}
OFFICIAL_SELECTION_ORDER = ["arc_agi_2", "arc_agi_1"]
OFFICIAL_D4_ORDER = [
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "anti_transpose",
]
COMBINED_D4_ORDER = ["rotate_90", "rotate_180", "flip_horizontal", "transpose"]
FEATURE_NAMES = [
    "demonstration_count_bin",
    "test_input_count_bin",
    "max_input_area_bin",
    "input_color_count_bin",
    "demonstration_shape_change",
]


@dataclass(frozen=True)
class Fixed64Build:
    manifest: dict[str, object]
    variant_files: dict[str, bytes]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _repo_file(root: Path, declared: str) -> Path:
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"path must be repository-relative: {declared}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes repository: {declared}") from error
    if not path.is_file():
        raise ValueError(f"file does not exist: {declared}")
    return path


def _domain_hash(*parts: object) -> str:
    encoded = b"\0".join(str(part).encode("utf-8") for part in parts)
    return hashlib.sha256(encoded).hexdigest()


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _input_grids(task: dict[str, Any]) -> Iterable[Grid]:
    for pair in task["train"]:
        yield pair["input"]
    for pair in task["test"]:
        yield pair["input"]


def _visible_grids(task: dict[str, Any]) -> Iterable[Grid]:
    for pair in task["train"]:
        yield pair["input"]
        yield pair["output"]
    for pair in task["test"]:
        yield pair["input"]


def _demo_bin(value: int) -> str:
    return str(value) if value <= 4 else "5+"


def _test_bin(value: int) -> str:
    return "1" if value == 1 else "2+"


def _area_bin(value: int) -> str:
    if value <= 100:
        return "1-100"
    if value <= 400:
        return "101-400"
    return "401-900"


def _color_bin(value: int) -> str:
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    return "7-10"


def _features(task: dict[str, Any]) -> tuple[dict[str, object], dict[str, object]]:
    input_grids = list(_input_grids(task))
    input_colors = sorted(
        {color for grid in input_grids for row in grid for color in row}
    )
    raw: dict[str, object] = {
        "demonstration_count": len(task["train"]),
        "test_input_count": len(task["test"]),
        "max_input_area": max(len(grid) * len(grid[0]) for grid in input_grids),
        "input_colors": input_colors,
        "input_color_count": len(input_colors),
        # Only dimensions are inspected; demonstration output cell values are
        # deliberately excluded from selection features.
        "demonstration_shape_change": any(
            _shape(pair["input"]) != _shape(pair["output"])
            for pair in task["train"]
        ),
    }
    binned: dict[str, object] = {
        "demonstration_count_bin": _demo_bin(int(raw["demonstration_count"])),
        "test_input_count_bin": _test_bin(int(raw["test_input_count"])),
        "max_input_area_bin": _area_bin(int(raw["max_input_area"])),
        "input_color_count_bin": _color_bin(int(raw["input_color_count"])),
        "demonstration_shape_change": bool(raw["demonstration_shape_change"]),
    }
    return raw, binned


def _stratum_key(features: dict[str, object]) -> str:
    return "|".join(str(features[name]) for name in FEATURE_NAMES)


def _largest_remainder_quotas(
    groups: dict[str, list[dict[str, Any]]],
    quota: int,
    *,
    domain: str,
    seed: int,
    benchmark_id: str,
) -> dict[str, int]:
    population = sum(len(items) for items in groups.values())
    if population < quota or quota <= 0:
        raise ValueError(f"invalid quota/population for {benchmark_id}")
    allocated = {
        key: quota * len(items) // population for key, items in groups.items()
    }
    remaining = quota - sum(allocated.values())
    ranked = sorted(
        groups,
        key=lambda key: (
            -(quota * len(groups[key]) % population),
            _domain_hash(domain, seed, "quota", benchmark_id, key),
            key,
        ),
    )
    for key in ranked[:remaining]:
        allocated[key] += 1
    if sum(allocated.values()) != quota:
        raise ValueError(f"largest-remainder quota failed: {benchmark_id}")
    if any(allocated[key] > len(groups[key]) for key in groups):
        raise ValueError(f"stratum quota exceeds population: {benchmark_id}")
    return allocated


def _select_benchmark(
    candidates: list[dict[str, Any]],
    quota: int,
    *,
    domain: str,
    seed: int,
    benchmark_id: str,
    excluded_task_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, object]], list[dict[str, object]]]:
    eligible = [item for item in candidates if item["task_id"] not in excluded_task_ids]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        groups[item["stratum_key"]].append(item)
    quotas = _largest_remainder_quotas(
        groups, quota, domain=domain, seed=seed, benchmark_id=benchmark_id
    )
    selected: list[dict[str, Any]] = []
    stratum_records: list[dict[str, object]] = []
    for key in sorted(groups):
        items = sorted(
            groups[key],
            key=lambda item: (item["rank_sha256"], item["task_id"]),
        )
        selected.extend(items[: quotas[key]])
        stratum_records.append(
            {
                "stratum_key": key,
                "population": len(items),
                "quota": quotas[key],
                "remainder_numerator": quota * len(items) % len(eligible),
                "quota_tie_sha256": _domain_hash(
                    domain, seed, "quota", benchmark_id, key
                ),
            }
        )
    selected.sort(key=lambda item: (item["rank_sha256"], item["task_id"]))
    selected_ids = {item["task_id"] for item in selected}
    quota_by_stratum = {item["stratum_key"]: item["quota"] for item in stratum_records}
    population_by_stratum = {
        item["stratum_key"]: item["population"] for item in stratum_records
    }
    inventory: list[dict[str, object]] = []
    for item in sorted(candidates, key=lambda candidate: candidate["task_id"]):
        duplicate_excluded = item["task_id"] in excluded_task_ids
        is_selected = item["task_id"] in selected_ids and not duplicate_excluded
        if duplicate_excluded:
            reason = "excluded_same_task_id_already_selected_for_arc_agi_2"
        elif is_selected:
            reason = "selected_within_frozen_stratum_quota"
        else:
            reason = "not_selected_within_frozen_stratum_quota"
        inventory.append(
            {
                "benchmark_generation": benchmark_id,
                "task_id": item["task_id"],
                "path": item["path"],
                "sha256": item["sha256"],
                "raw_features": item["raw_features"],
                "selection_features": item["selection_features"],
                "stratum_key": item["stratum_key"],
                "stratum_population": (
                    None if duplicate_excluded else population_by_stratum[item["stratum_key"]]
                ),
                "stratum_quota": (
                    None if duplicate_excluded else quota_by_stratum[item["stratum_key"]]
                ),
                "rank_sha256": item["rank_sha256"],
                "eligible_for_selection": not duplicate_excluded,
                "selected": is_selected,
                "decision_reason": reason,
            }
        )
    return selected, inventory, stratum_records


def _global_color_maps(domain: str, seeds: list[int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        for retry_salt in range(1000):
            values = sorted(
                range(1, 10),
                key=lambda color: _domain_hash(
                    domain, seed, "color", retry_salt, color
                ),
            )
            signature = tuple(values)
            if signature == tuple(range(1, 10)) or signature in seen:
                continue
            mapping = {0: 0, **{key: value for key, value in zip(range(1, 10), values)}}
            color_transform(mapping)
            seen.add(signature)
            records.append(
                {
                    "seed": seed,
                    "retry_salt": retry_salt,
                    "mapping": {str(key): value for key, value in mapping.items()},
                    "mapping_sha256": canonical_sha256(mapping),
                }
            )
            break
        else:  # pragma: no cover - cryptographic ordering makes this unreachable
            raise ValueError(f"could not derive color mapping for seed {seed}")
    if len(records) != 4 or len(seen) != 4:
        raise ValueError("four distinct non-identity color permutations are required")
    return records


def _compose(first: Callable[[Grid], Grid], second: Callable[[Grid], Grid]) -> Callable[[Grid], Grid]:
    return lambda grid: second(first(grid))


def _restore_task(
    transformed: dict[str, Any],
    inverse_grid: Callable[[Grid], Grid],
    *,
    train_order: Sequence[int] | None,
    test_order: Sequence[int] | None,
) -> dict[str, Any]:
    reordered: dict[str, Any] = {
        "train": (
            restore_test_order(transformed["train"], train_order)
            if train_order is not None
            else transformed["train"]
        ),
        "test": (
            restore_test_order(transformed["test"], test_order)
            if test_order is not None
            else transformed["test"]
        ),
    }
    if "name" in transformed:
        reordered["name"] = transformed["name"]
    return transform_task(reordered, inverse_grid)


def _prediction_witness(task: dict[str, Any]) -> list[dict[str, Grid]]:
    return [
        {
            "attempt_1": [row[:] for row in pair["input"]],
            "attempt_2": [list(reversed(row)) for row in pair["input"]],
        }
        for pair in task["test"]
    ]


def _apply_prediction_variant(
    predictions: list[dict[str, Grid]],
    grid_transform: Callable[[Grid], Grid],
    test_order: Sequence[int] | None,
) -> list[dict[str, Grid]]:
    transformed = transform_predictions(predictions, grid_transform)
    return (
        [transformed[index] for index in test_order]
        if test_order is not None
        else transformed
    )


def _restore_predictions(
    predictions: list[dict[str, Grid]],
    inverse_grid: Callable[[Grid], Grid],
    test_order: Sequence[int] | None,
) -> list[dict[str, Grid]]:
    inverse_transformed = transform_predictions(predictions, inverse_grid)
    return (
        restore_test_order(inverse_transformed, test_order)
        if test_order is not None
        else inverse_transformed
    )


def _variant(
    *,
    task: dict[str, Any],
    benchmark_id: str,
    task_id: str,
    variant_id: str,
    category: str,
    forward_grid: Callable[[Grid], Grid],
    inverse_grid: Callable[[Grid], Grid],
    specification: dict[str, object],
    train_order: Sequence[int] | None = None,
    test_order: Sequence[int] | None = None,
) -> tuple[dict[str, object], bytes]:
    transformed = transform_task(
        task,
        forward_grid,
        train_order=train_order,
        test_order=test_order,
    )
    if any("output" in pair for pair in transformed["test"]):
        raise ValueError("IsoARC challenge variant contains a hidden test output")
    restored = _restore_task(
        transformed,
        inverse_grid,
        train_order=train_order,
        test_order=test_order,
    )
    if canonical_json_bytes(restored) != canonical_json_bytes(task):
        raise ValueError(f"task round-trip failed: {benchmark_id}:{task_id}:{variant_id}")
    witness = _prediction_witness(task)
    transformed_predictions = _apply_prediction_variant(
        witness, forward_grid, test_order
    )
    restored_predictions = _restore_predictions(
        transformed_predictions, inverse_grid, test_order
    )
    if restored_predictions != witness:
        raise ValueError(
            f"prediction round-trip failed: {benchmark_id}:{task_id}:{variant_id}"
        )
    payload = pretty_json_bytes(transformed)
    path = f"variants/{benchmark_id}/{task_id}/{variant_id}.json"
    record: dict[str, object] = {
        "variant_id": variant_id,
        "category": category,
        "specification": specification,
        "train_order": None if train_order is None else list(train_order),
        "test_order": None if test_order is None else list(test_order),
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "test_output_fields_present": 0,
        "task_round_trip_passed": True,
        "prediction_round_trip_passed": True,
    }
    return record, payload


def _feature_distribution(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name in FEATURE_NAMES:
        counts = Counter(str(item["selection_features"][name]) for item in items)
        result[name] = dict(sorted(counts.items()))
    return result


def build_fixed64_design(root: Path, config_path: Path) -> Fixed64Build:
    root = root.resolve()
    config_path = config_path.resolve()
    config = _load_object(config_path)
    if config.get("profile_id") != OFFICIAL_PROFILE_ID:
        raise ValueError("unexpected fixed64 profile_id")
    if config.get("protocol_status") != "frozen-inputs":
        raise ValueError("fixed64 config must declare frozen-inputs")
    selection = config.get("selection")
    variant_design = config.get("variant_design")
    if not isinstance(selection, dict) or not isinstance(variant_design, dict):
        raise ValueError("selection and variant_design must be objects")
    if selection.get("quota_by_benchmark") != OFFICIAL_QUOTAS:
        raise ValueError("fixed64 quotas must be 32 ARC-AGI-1 and 32 ARC-AGI-2")
    if selection.get("visible_features") != FEATURE_NAMES:
        raise ValueError("input-visible feature roster changed")
    if selection.get("benchmark_selection_order") != OFFICIAL_SELECTION_ORDER:
        raise ValueError("benchmark selection order changed")
    if selection.get("algorithm") != "exact-stratum-largest-remainder-v1":
        raise ValueError("selection algorithm changed")
    seed = selection.get("public_seed")
    domain = selection.get("domain")
    if type(seed) is not int or not isinstance(domain, str) or not domain:
        raise ValueError("selection seed/domain are invalid")
    if variant_design.get("d4_order") != OFFICIAL_D4_ORDER:
        raise ValueError("D4 order changed")
    if variant_design.get("combined_d4_color_assignment") != "balanced-8x4-cyclic-latin-v1":
        raise ValueError("combined D4/color assignment changed")
    if variant_design.get("demonstration_permutations_when_at_least_three") != 2:
        raise ValueError("two demonstration permutations are required")
    if (
        variant_design.get("test_order_policy")
        != "cyclic-left-when-at-least-two-plus-reverse-when-at-least-three"
    ):
        raise ValueError("test-order policy changed")
    color_seeds = variant_design.get("fixed_zero_color_seeds")
    if (
        not isinstance(color_seeds, list)
        or len(color_seeds) != 4
        or any(type(value) is not int for value in color_seeds)
        or len(set(color_seeds)) != 4
    ):
        raise ValueError("four distinct integer color seeds are required")

    challenge_declaration = config.get("challenge_manifest")
    if not isinstance(challenge_declaration, dict):
        raise ValueError("challenge_manifest must be an object")
    challenge_path = _repo_file(root, str(challenge_declaration.get("path")))
    if sha256_file(challenge_path) != challenge_declaration.get("sha256"):
        raise ValueError("challenge manifest file hash mismatch")
    challenge = _load_object(challenge_path)
    challenge_payload = dict(challenge)
    declared_challenge_digest = challenge_payload.pop("challenge_manifest_sha256", None)
    if declared_challenge_digest != challenge_declaration.get(
        "challenge_manifest_sha256"
    ) or canonical_sha256(challenge_payload) != declared_challenge_digest:
        raise ValueError("challenge manifest canonical digest mismatch")

    candidates_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    tasks_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for benchmark_id in OFFICIAL_QUOTAS:
        view = challenge.get("views", {}).get(benchmark_id)
        if not isinstance(view, dict):
            raise ValueError(f"missing challenge view: {benchmark_id}")
        records = view.get("files")
        if not isinstance(records, list) or len(records) != view.get("task_count"):
            raise ValueError(f"challenge denominator mismatch: {benchmark_id}")
        candidates: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"invalid challenge record: {benchmark_id}")
            declared_task_path = (
                challenge_path.parent / str(record.get("path"))
            ).relative_to(root).as_posix()
            task_path = _repo_file(root, declared_task_path)
            if sha256_file(task_path) != record.get("sha256"):
                raise ValueError(f"challenge task hash mismatch: {task_path}")
            task = load_task(task_path, require_test_outputs=False)
            if any("output" in pair for pair in task["test"]):
                raise ValueError(f"hidden test output present: {task_path}")
            task_id = str(record.get("task_id"))
            raw, feature_bins = _features(task)
            candidate = {
                "benchmark_generation": benchmark_id,
                "task_id": task_id,
                "path": task_path.relative_to(root).as_posix(),
                "sha256": str(record["sha256"]),
                "raw_features": raw,
                "selection_features": feature_bins,
                "stratum_key": _stratum_key(feature_bins),
                "rank_sha256": _domain_hash(
                    domain, seed, "rank", benchmark_id, task_id
                ),
            }
            candidates.append(candidate)
            tasks_by_identity[(benchmark_id, task_id)] = task
        if len({item["task_id"] for item in candidates}) != len(candidates):
            raise ValueError(f"duplicate task ID within challenge view: {benchmark_id}")
        candidates_by_benchmark[benchmark_id] = candidates

    selections: dict[str, list[dict[str, Any]]] = {}
    candidate_inventory: list[dict[str, object]] = []
    stratum_quotas: dict[str, list[dict[str, object]]] = {}
    selected_task_ids: set[str] = set()
    for benchmark_id in OFFICIAL_SELECTION_ORDER:
        selected, inventory, strata = _select_benchmark(
            candidates_by_benchmark[benchmark_id],
            OFFICIAL_QUOTAS[benchmark_id],
            domain=domain,
            seed=seed,
            benchmark_id=benchmark_id,
            excluded_task_ids=(selected_task_ids if benchmark_id == "arc_agi_1" else set()),
        )
        selections[benchmark_id] = selected
        candidate_inventory.extend(inventory)
        stratum_quotas[benchmark_id] = strata
        selected_task_ids.update(item["task_id"] for item in selected)
    if len(selected_task_ids) != 64:
        raise ValueError("fixed64 base task IDs are not globally unique")

    color_maps = _global_color_maps(domain, color_seeds)
    parsed_color_maps = [
        {int(key): int(value) for key, value in record["mapping"].items()}
        for record in color_maps
    ]
    assignments: list[dict[str, object]] = []
    variant_files: dict[str, bytes] = {}
    flat_variant_records: list[dict[str, object]] = []
    latin_counts_by_benchmark: dict[str, Counter[str]] = {
        benchmark_id: Counter() for benchmark_id in OFFICIAL_QUOTAS
    }
    selected_feature_distributions: dict[str, object] = {}
    for benchmark_id in OFFICIAL_SELECTION_ORDER:
        selected = selections[benchmark_id]
        selected_feature_distributions[benchmark_id] = _feature_distribution(selected)
        for local_index, candidate in enumerate(selected):
            task_id = candidate["task_id"]
            task = tasks_by_identity[(benchmark_id, task_id)]
            task_variants: list[dict[str, object]] = []

            def add_variant(**kwargs: Any) -> None:
                record, payload = _variant(
                    task=task,
                    benchmark_id=benchmark_id,
                    task_id=task_id,
                    **kwargs,
                )
                if record["path"] in variant_files:
                    raise ValueError(f"duplicate variant path: {record['path']}")
                variant_files[str(record["path"])] = payload
                task_variants.append(record)
                flat_variant_records.append(
                    {
                        "benchmark_generation": benchmark_id,
                        "task_id": task_id,
                        **record,
                    }
                )

            for d4_name in OFFICIAL_D4_ORDER:
                add_variant(
                    variant_id=f"d4-{d4_name}",
                    category="d4_only",
                    forward_grid=D4[d4_name],
                    inverse_grid=D4[INVERSE_D4[d4_name]],
                    specification={"d4": d4_name},
                )
            for color_index, mapping in enumerate(parsed_color_maps):
                add_variant(
                    variant_id=f"color-{color_index}",
                    category="color_only",
                    forward_grid=color_transform(mapping),
                    inverse_grid=color_transform(inverse_color_mapping(mapping)),
                    specification={
                        "color_index": color_index,
                        "color_seed": color_maps[color_index]["seed"],
                        "mapping_sha256": color_maps[color_index]["mapping_sha256"],
                    },
                )
            latin_row = local_index % 4
            for d4_slot, d4_name in enumerate(COMBINED_D4_ORDER):
                color_index = (latin_row + d4_slot) % 4
                mapping = parsed_color_maps[color_index]
                forward = _compose(D4[d4_name], color_transform(mapping))
                inverse = _compose(
                    color_transform(inverse_color_mapping(mapping)),
                    D4[INVERSE_D4[d4_name]],
                )
                latin_key = f"{d4_name}|color-{color_index}"
                latin_counts_by_benchmark[benchmark_id][latin_key] += 1
                add_variant(
                    variant_id=f"combo-{d4_name}-color-{color_index}",
                    category="d4_plus_color",
                    forward_grid=forward,
                    inverse_grid=inverse,
                    specification={
                        "d4": d4_name,
                        "d4_slot": d4_slot,
                        "latin_row": latin_row,
                        "color_index": color_index,
                        "color_seed": color_maps[color_index]["seed"],
                        "mapping_sha256": color_maps[color_index]["mapping_sha256"],
                    },
                )
            if len(task["train"]) >= 3:
                demo_orders = [
                    list(range(1, len(task["train"]))) + [0],
                    list(reversed(range(len(task["train"])))),
                ]
                if demo_orders[0] == demo_orders[1]:
                    raise ValueError("demonstration permutations are not distinct")
                for name, order in zip(("cyclic-left", "reverse"), demo_orders):
                    add_variant(
                        variant_id=f"demo-order-{name}",
                        category="demonstration_order",
                        forward_grid=D4["identity"],
                        inverse_grid=D4["identity"],
                        specification={"permutation": name},
                        train_order=order,
                    )
            if len(task["test"]) >= 2:
                test_orders = [
                    (
                        "cyclic-left",
                        list(range(1, len(task["test"]))) + [0],
                    )
                ]
                if len(task["test"]) >= 3:
                    test_orders.append(
                        ("reverse", list(reversed(range(len(task["test"])))))
                    )
                for name, order in test_orders:
                    add_variant(
                        variant_id=f"test-order-{name}",
                        category="test_order",
                        forward_grid=D4["identity"],
                        inverse_grid=D4["identity"],
                        specification={"permutation": name},
                        test_order=order,
                    )
            expected_variant_count = (
                16
                + (2 if len(task["train"]) >= 3 else 0)
                + (1 if len(task["test"]) >= 2 else 0)
                + (1 if len(task["test"]) >= 3 else 0)
            )
            if len(task_variants) != expected_variant_count:
                raise ValueError("logical variant count does not match frozen formula")
            assignments.append(
                {
                    "benchmark_generation": benchmark_id,
                    "task_id": task_id,
                    "local_selected_index": local_index,
                    "source_challenge_path": candidate["path"],
                    "source_challenge_sha256": candidate["sha256"],
                    "raw_features": candidate["raw_features"],
                    "selection_features": candidate["selection_features"],
                    "stratum_key": candidate["stratum_key"],
                    "rank_sha256": candidate["rank_sha256"],
                    "latin_row": latin_row,
                    "logical_variant_count": len(task_variants),
                    "variants": task_variants,
                }
            )

    expected_latin_keys = {
        f"{d4_name}|color-{color_index}"
        for d4_name in COMBINED_D4_ORDER
        for color_index in range(4)
    }
    for benchmark_id, counts in latin_counts_by_benchmark.items():
        if set(counts) != expected_latin_keys or set(counts.values()) != {8}:
            raise ValueError(f"Latin balance is not exact: {benchmark_id}")
    flat_variant_records.sort(
        key=lambda item: (
            item["benchmark_generation"],
            item["task_id"],
            item["variant_id"],
        )
    )
    candidate_inventory.sort(
        key=lambda item: (item["benchmark_generation"], item["task_id"])
    )
    selected_identity = [
        f"{item['benchmark_generation']}:{item['task_id']}" for item in assignments
    ]
    unqualified_selected_ids = [str(item["task_id"]) for item in assignments]
    category_counts = Counter(
        str(record["category"]) for record in flat_variant_records
    )
    variant_inventory = [
        {
            "path": record["path"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
        for record in flat_variant_records
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": OFFICIAL_PROFILE_ID,
        "freeze_status": "frozen",
        "protocol_status": "frozen-inputs",
        "scope": "64-base-task label-free public IsoARC curve and robustness design",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "source_manifest": {
            "path": challenge_path.relative_to(root).as_posix(),
            "sha256": sha256_file(challenge_path),
            "challenge_manifest_sha256": declared_challenge_digest,
            "source_task_count": sum(
                len(candidates) for candidates in candidates_by_benchmark.values()
            ),
            "test_output_fields_read_by_selector": 0,
        },
        "feature_contract": {
            "features": FEATURE_NAMES,
            "demonstration_count_bins": ["1", "2", "3", "4", "5+"],
            "test_input_count_bins": ["1", "2+"],
            "max_input_area_bins": ["1-100", "101-400", "401-900"],
            "input_color_count_bins": ["1-3", "4-6", "7-10"],
            "demonstration_shape_change": "compare train input/output dimensions only",
            "cell_values_used": "train inputs and test inputs only",
        },
        "selection_contract": {
            "algorithm": selection["algorithm"],
            "domain": domain,
            "public_seed": seed,
            "quota_by_benchmark": OFFICIAL_QUOTAS,
            "benchmark_selection_order": OFFICIAL_SELECTION_ORDER,
            "cross_benchmark_duplicate_policy": selection[
                "cross_benchmark_duplicate_policy"
            ],
            "rank_domain_fields": [
                "domain",
                "public_seed",
                "rank",
                "benchmark_generation",
                "task_id",
            ],
            "rank_excludes_task_content_hashes": True,
        },
        "stratum_quotas": stratum_quotas,
        "candidate_inventory": candidate_inventory,
        "selected_feature_distributions": selected_feature_distributions,
        "color_maps": color_maps,
        "transform_contract": {
            "d4_order": OFFICIAL_D4_ORDER,
            "combined_d4_order": COMBINED_D4_ORDER,
            "color_map_count": 4,
            "color_domain": list(range(10)),
            "color_zero_fixed": True,
            "logical_variant_count_formula": (
                "16 + 2*I(demos>=3) + I(tests>=2) + I(tests>=3)"
            ),
            "symmetric_duplicate_policy": "retain every logical variant ID",
        },
        "assignments": assignments,
        "balance_tables": {
            "latin_pair_counts_by_benchmark": {
                benchmark_id: dict(sorted(counts.items()))
                for benchmark_id, counts in latin_counts_by_benchmark.items()
            },
            "logical_variant_category_counts": dict(sorted(category_counts.items())),
        },
        "digests": {
            "candidate_inventory_sha256": canonical_sha256(candidate_inventory),
            "selection_sha256": canonical_sha256(selected_identity),
            "assignment_sha256": canonical_sha256(assignments),
            "variant_inventory_sha256": canonical_sha256(variant_inventory),
        },
        "summary": {
            "source_task_count": len(candidate_inventory),
            "selected_base_task_count": len(assignments),
            "selected_by_benchmark": dict(
                sorted(
                    Counter(
                        str(item["benchmark_generation"]) for item in assignments
                    ).items()
                )
            ),
            "unique_task_id_count": len(set(unqualified_selected_ids)),
            "logical_variant_count": len(flat_variant_records),
            "logical_variant_category_counts": dict(sorted(category_counts.items())),
            "challenge_test_output_field_count": 0,
        },
        "contract_checks": {
            "all_source_challenge_hashes_verified": True,
            "selector_opened_labeled_files": 0,
            "all_variant_files_label_free": True,
            "all_task_round_trips": True,
            "all_prediction_round_trips": True,
            "color_bijections_valid": True,
            "latin_balance_exact": True,
            "test_order_restoration": True,
            "cross_benchmark_task_ids_unique": len(set(unqualified_selected_ids)) == 64,
        },
        "variant_file_inventory": variant_inventory,
        "claim_boundary": config["claim_boundary"],
        "limitations": [
            "The public base tasks and labels were historically accessible; this diagnostic is neither private nor label-naive.",
            "The 32/32 strata are reported separately and are not an unweighted estimate for the unequal 400/120 source populations.",
            "IsoARC variants remain clustered by base task and never increase the independent task denominator.",
            "This manifest freezes diagnostic inputs and transformations; it contains no solver prediction or accuracy result."
        ],
    }
    manifest["fixed64_manifest_sha256"] = canonical_sha256(manifest)
    return Fixed64Build(manifest=manifest, variant_files=variant_files)


def build_fixed64_manifest(root: Path, config_path: Path) -> dict[str, object]:
    """Compatibility helper returning only the deterministic manifest."""

    return build_fixed64_design(root, config_path).manifest


def audit_labeled_references(
    root: Path,
    design_manifest: dict[str, object],
) -> dict[str, object]:
    """Audit hidden-label mutation and labeled round trips after design freeze.

    This function is for the independent auditor/scorer process. Selection code
    never calls it and accepts no labeled source path.
    """

    root = root.resolve()
    source_declaration = design_manifest.get("source_manifest")
    if not isinstance(source_declaration, dict):
        raise ValueError("design source_manifest is missing")
    challenge_path = _repo_file(root, str(source_declaration.get("path")))
    if sha256_file(challenge_path) != source_declaration.get("sha256"):
        raise ValueError("design challenge manifest hash mismatch")
    challenge = _load_object(challenge_path)
    source_records: dict[tuple[str, str], dict[str, Any]] = {}
    labeled_tasks: dict[tuple[str, str], dict[str, Any]] = {}
    mutation_stable_count = 0
    visible_tree_rebuilds: dict[str, list[dict[str, object]]] = defaultdict(list)
    scorer_checks: dict[str, object] = {}
    for benchmark_id in OFFICIAL_QUOTAS:
        view = challenge["views"][benchmark_id]
        answers_original: dict[str, list[Grid]] = {}
        answers_mutated: dict[str, list[Grid]] = {}
        exact_predictions: dict[str, list[dict[str, Grid]]] = {}
        for record in view["files"]:
            task_id = str(record["task_id"])
            source_path = _repo_file(root, str(record.get("source_path")))
            if sha256_file(source_path) != record.get("source_sha256"):
                raise ValueError(f"labeled source hash mismatch: {source_path}")
            labeled = load_task(source_path)
            mutated = copy.deepcopy(labeled)
            for pair in mutated["test"]:
                pair["output"] = [
                    [(cell + 1) % 10 for cell in row] for row in pair["output"]
                ]
            original_challenge = challenge_only(labeled)
            mutated_challenge = challenge_only(mutated)
            original_payload = pretty_json_bytes(original_challenge)
            mutated_payload = pretty_json_bytes(mutated_challenge)
            if original_payload != mutated_payload:
                raise ValueError(f"hidden-label mutation changed challenge bytes: {task_id}")
            challenge_file = _repo_file(
                root,
                (challenge_path.parent / str(record["path"])).relative_to(root).as_posix(),
            )
            if challenge_file.read_bytes() != original_payload:
                raise ValueError(f"persisted challenge differs from stripped source: {task_id}")
            mutation_stable_count += 1
            visible_tree_rebuilds[benchmark_id].append(
                {
                    "path": record["path"],
                    "sha256": hashlib.sha256(mutated_payload).hexdigest(),
                    "bytes": len(mutated_payload),
                }
            )
            source_records[(benchmark_id, task_id)] = record
            labeled_tasks[(benchmark_id, task_id)] = labeled
            answers_original[task_id] = [pair["output"] for pair in labeled["test"]]
            answers_mutated[task_id] = [pair["output"] for pair in mutated["test"]]
            exact_predictions[task_id] = [
                {
                    "attempt_1": pair["output"],
                    "attempt_2": pair["output"],
                }
                for pair in labeled["test"]
            ]
        original_score = score_predictions(
            exact_predictions, answers_original, top_k=2, source="auditor-sentinel"
        )
        mutated_score = score_predictions(
            exact_predictions, answers_mutated, top_k=2, source="auditor-sentinel"
        )
        if original_score.outputs_exact != original_score.outputs_total:
            raise ValueError("scorer sentinel is not exact on original labels")
        if mutated_score.outputs_exact != 0:
            raise ValueError("scorer sentinel did not distinguish mutated labels")
        scorer_checks[benchmark_id] = {
            "original_outputs_exact": original_score.outputs_exact,
            "original_outputs_total": original_score.outputs_total,
            "mutated_outputs_exact": mutated_score.outputs_exact,
            "label_sensitivity_passed": True,
        }

    color_maps = [
        {int(key): int(value) for key, value in record["mapping"].items()}
        for record in design_manifest["color_maps"]
    ]
    labeled_task_round_trips = 0
    labeled_prediction_round_trips = 0
    for assignment in design_manifest["assignments"]:
        benchmark_id = str(assignment["benchmark_generation"])
        task_id = str(assignment["task_id"])
        labeled = labeled_tasks[(benchmark_id, task_id)]
        labeled_predictions = [
            {
                "attempt_1": pair["output"],
                "attempt_2": [row[:] for row in pair["output"]],
            }
            for pair in labeled["test"]
        ]
        for variant in assignment["variants"]:
            category = variant["category"]
            specification = variant["specification"]
            if category == "d4_only":
                d4_name = specification["d4"]
                forward = D4[d4_name]
                inverse = D4[INVERSE_D4[d4_name]]
            elif category == "color_only":
                mapping = color_maps[int(specification["color_index"])]
                forward = color_transform(mapping)
                inverse = color_transform(inverse_color_mapping(mapping))
            elif category == "d4_plus_color":
                d4_name = specification["d4"]
                mapping = color_maps[int(specification["color_index"])]
                forward = _compose(D4[d4_name], color_transform(mapping))
                inverse = _compose(
                    color_transform(inverse_color_mapping(mapping)),
                    D4[INVERSE_D4[d4_name]],
                )
            else:
                forward = D4["identity"]
                inverse = D4["identity"]
            train_order = variant["train_order"]
            test_order = variant["test_order"]
            transformed = transform_task(
                labeled,
                forward,
                train_order=train_order,
                test_order=test_order,
            )
            restored = _restore_task(
                transformed,
                inverse,
                train_order=train_order,
                test_order=test_order,
            )
            if canonical_json_bytes(restored) != canonical_json_bytes(labeled):
                raise ValueError(
                    f"labeled task round-trip failed: {benchmark_id}:{task_id}:"
                    f"{variant['variant_id']}"
                )
            labeled_task_round_trips += 1
            forward_predictions = _apply_prediction_variant(
                labeled_predictions, forward, test_order
            )
            restored_predictions = _restore_predictions(
                forward_predictions, inverse, test_order
            )
            if restored_predictions != labeled_predictions:
                raise ValueError(
                    f"labeled prediction round-trip failed: {benchmark_id}:{task_id}:"
                    f"{variant['variant_id']}"
                )
            labeled_prediction_round_trips += 1

    expected_source_count = int(design_manifest["summary"]["source_task_count"])
    expected_variant_count = int(design_manifest["summary"]["logical_variant_count"])
    if mutation_stable_count != expected_source_count:
        raise ValueError("hidden-label mutation audit denominator mismatch")
    if labeled_task_round_trips != expected_variant_count:
        raise ValueError("labeled task round-trip denominator mismatch")
    if labeled_prediction_round_trips != expected_variant_count:
        raise ValueError("labeled prediction round-trip denominator mismatch")
    return {
        "labeled_source_file_count": len(source_records),
        "hidden_label_mutation_task_count": mutation_stable_count,
        "hidden_label_mutation_stable": True,
        "labeled_task_round_trip_count": labeled_task_round_trips,
        "labeled_prediction_round_trip_count": labeled_prediction_round_trips,
        "all_labeled_task_round_trips": True,
        "all_labeled_prediction_round_trips": True,
        "scorer_label_sensitivity": scorer_checks,
        "scorer_label_sensitivity_passed": True,
        "selection_or_assignment_modified_by_auditor": False,
    }


__all__ = [
    "COMBINED_D4_ORDER",
    "FEATURE_NAMES",
    "Fixed64Build",
    "OFFICIAL_D4_ORDER",
    "OFFICIAL_PROFILE_ID",
    "OFFICIAL_QUOTAS",
    "build_fixed64_design",
    "build_fixed64_manifest",
    "audit_labeled_references",
    "canonical_sha256",
    "sha256_file",
]
