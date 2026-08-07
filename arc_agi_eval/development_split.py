"""Training-only ARC development clustering and deterministic split assignment.

The declared equivalence relation uses complete labeled training tasks.  Two
tasks are merged only after replay-verifying one global D4 transform, one global
color bijection, and independent train/test example permutations.  Candidate
signatures are only a search optimization and never sufficient for a merge.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .isoarc import D4
from .overlap import normalized_task
from .validation import Grid, load_task, validate_task


DEFAULT_PUBLIC_SEED = 20260806
DEFAULT_SPLIT_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("dev-build", 70),
    ("dev-select", 15),
    ("dev-audit", 15),
)
DEFAULT_MAX_SEARCH_STATES = 100_000
SOURCE_NAMES = ("arc-agi-1-training", "arc-agi-2-training")


def canonical_json_bytes(value: object) -> bytes:
    """Return the stable JSON encoding used by every manifest digest."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_overlap_id_ledger(
    path: str | Path,
    *,
    expected_sha256: str,
    report_reference: str,
) -> dict[str, object]:
    """Load only the ID ledger from an already-persisted overlap audit.

    This function never follows the split paths named by the report and never
    opens ARC evaluation tasks. The locked audit is annotation input only.
    """

    audit_path = Path(path).resolve()
    observed_sha256 = sha256_file(audit_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "overlap audit SHA-256 mismatch: "
            f"expected {expected_sha256}, found {observed_sha256}"
        )
    value = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "passed":
        raise ValueError("overlap audit must be a passed JSON record")
    if value.get("scope") != "cross-benchmark-labeled-task-overlap":
        raise ValueError("unexpected overlap audit scope")
    overlap = value.get("overlap")
    if not isinstance(overlap, dict):
        raise ValueError("overlap audit is missing its overlap object")
    task_ids = overlap.get("id_overlap")
    if (
        not isinstance(task_ids, list)
        or any(not isinstance(task_id, str) for task_id in task_ids)
        or task_ids != sorted(task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError("overlap ID ledger must be a sorted unique string array")
    if overlap.get("id_overlap_count") != len(task_ids):
        raise ValueError("overlap ID count does not match its ledger")
    ledger_sha256 = fingerprint(task_ids)
    if overlap.get("overlap_digest") != ledger_sha256:
        raise ValueError("overlap ID ledger digest mismatch")
    test_io_exact = overlap.get("test_io_exact")
    if not isinstance(test_io_exact, list) or test_io_exact != task_ids:
        raise ValueError("expected every overlap ID to have exact test I/O")
    return {
        "report_reference": report_reference,
        "report_file_sha256": observed_sha256,
        "report_run_id": value.get("run_id"),
        "report_status": value.get("status"),
        "ledger_kind": (
            "ARC-AGI-1-public-evaluation-ID overlap with ARC-AGI-2-training"
        ),
        "task_id_count": len(task_ids),
        "task_id_ledger_sha256": ledger_sha256,
        "task_ids": task_ids,
        "semantic_labeled_exact_count": overlap.get(
            "semantic_labeled_exact_count"
        ),
        "test_io_exact_count": overlap.get("test_io_exact_count"),
        "direct_evaluation_files_opened": 0,
    }


@dataclass(frozen=True)
class TaskRecord:
    record_id: str
    source: str
    task_id: str
    relative_path: str
    source_file_sha256: str
    full_task_content_sha256: str
    task: dict[str, Any]

    def to_manifest(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "task_id": self.task_id,
            "relative_path": self.relative_path,
            "source_file_sha256": self.source_file_sha256,
            "full_task_content_sha256": self.full_task_content_sha256,
        }


@dataclass(frozen=True)
class TrainingSource:
    name: str
    path: Path
    records: tuple[TaskRecord, ...]
    tree_sha256: str

    def to_manifest(self) -> dict[str, object]:
        return {
            "dataset_role": "development-training-only",
            "logical_source": self.name,
            "required_directory_name": "training",
            "task_count": len(self.records),
            "tree_sha256": self.tree_sha256,
            "tree_digest_scheme": (
                "sha256(canonical-json(sorted[{relative_path,file_sha256}]))"
            ),
        }


@dataclass(frozen=True)
class IsomorphismWitness:
    d4_transform: str
    color_mapping: dict[int, int]
    train_permutation: tuple[int, ...]
    test_permutation: tuple[int, ...]

    def to_manifest(self) -> dict[str, object]:
        return {
            "d4_transform": self.d4_transform,
            "color_mapping": {
                str(source): target
                for source, target in sorted(self.color_mapping.items())
            },
            "train_permutation": list(self.train_permutation),
            "test_permutation": list(self.test_permutation),
        }


@dataclass(frozen=True)
class IsomorphismSearchResult:
    status: str
    witness: IsomorphismWitness | None
    explored_states: int


@dataclass(frozen=True)
class ClusterResult:
    groups: tuple[tuple[TaskRecord, ...], ...]
    verified_edges: tuple[dict[str, object], ...]
    inconclusive_pairs: tuple[dict[str, object], ...]
    candidate_bucket_count: int
    candidate_pair_count: int
    explored_search_states: int


class _SearchLimitReached(RuntimeError):
    pass


class _SearchBudget:
    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_search_states must be a positive integer")
        self.limit = limit
        self.states = 0

    def tick(self) -> None:
        if self.states >= self.limit:
            raise _SearchLimitReached
        self.states += 1


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _full_task_content(task: dict[str, Any]) -> dict[str, object]:
    return normalized_task(
        task,
        normalize_train_order=True,
        normalize_test_order=True,
        include_test_outputs=True,
    )


def _grid_color_counts(grid: Grid) -> tuple[int, ...]:
    return tuple(sorted(Counter(cell for row in grid for cell in row).values()))


def _d4_invariant_grid_signature(grid: Grid) -> tuple[object, ...]:
    height = len(grid)
    width = len(grid[0])
    return (min(height, width), max(height, width), _grid_color_counts(grid))


def _oriented_grid_signature(grid: Grid) -> tuple[object, ...]:
    return (len(grid), len(grid[0]), _grid_color_counts(grid))


def _example_signature(
    example: dict[str, Grid], *, d4_invariant: bool
) -> tuple[object, ...]:
    grid_signature = (
        _d4_invariant_grid_signature
        if d4_invariant
        else _oriented_grid_signature
    )
    if "output" in example:
        output: tuple[object, ...] = ("present", grid_signature(example["output"]))
    else:
        output = ("missing",)
    return (grid_signature(example["input"]), output)


def task_candidate_signature(task: dict[str, Any]) -> tuple[object, ...]:
    """Necessary D4/color invariant used only to limit verified comparisons."""

    return (
        tuple(
            sorted(
                _example_signature(example, d4_invariant=True)
                for example in task["train"]
            )
        ),
        tuple(
            sorted(
                _example_signature(example, d4_invariant=True)
                for example in task["test"]
            )
        ),
    )


def _transform_example(
    example: dict[str, Grid], transform_name: str
) -> dict[str, Grid]:
    transform = D4[transform_name]
    result = {"input": transform(example["input"])}
    if "output" in example:
        result["output"] = transform(example["output"])
    return result


def _extend_color_bijection(
    left: dict[str, Grid],
    right: dict[str, Grid],
    mapping: dict[int, int],
    inverse: dict[int, int],
) -> tuple[dict[int, int], dict[int, int]] | None:
    if ("output" in left) != ("output" in right):
        return None
    updated = dict(mapping)
    updated_inverse = dict(inverse)
    for field in ("input", "output"):
        if field not in left:
            continue
        left_grid = left[field]
        right_grid = right[field]
        if len(left_grid) != len(right_grid):
            return None
        if len(left_grid[0]) != len(right_grid[0]):
            return None
        for left_row, right_row in zip(left_grid, right_grid):
            for source_color, target_color in zip(left_row, right_row):
                existing_target = updated.get(source_color)
                existing_source = updated_inverse.get(target_color)
                if existing_target is not None and existing_target != target_color:
                    return None
                if existing_source is not None and existing_source != source_color:
                    return None
                updated[source_color] = target_color
                updated_inverse[target_color] = source_color
    return updated, updated_inverse


def _search_one_transform(
    left: dict[str, Any],
    right: dict[str, Any],
    transform_name: str,
    budget: _SearchBudget,
) -> IsomorphismWitness | None:
    transformed: dict[str, list[dict[str, Grid]]] = {
        split: [_transform_example(example, transform_name) for example in left[split]]
        for split in ("train", "test")
    }
    target: dict[str, list[dict[str, Grid]]] = {
        split: list(right[split]) for split in ("train", "test")
    }
    if any(len(transformed[split]) != len(target[split]) for split in transformed):
        return None

    target_signatures = {
        split: [
            _example_signature(example, d4_invariant=False)
            for example in target[split]
        ]
        for split in ("train", "test")
    }
    slots: list[tuple[str, int, tuple[int, ...]]] = []
    for split in ("train", "test"):
        for left_index, example in enumerate(transformed[split]):
            signature = _example_signature(example, d4_invariant=False)
            candidates = tuple(
                target_index
                for target_index, target_signature in enumerate(
                    target_signatures[split]
                )
                if target_signature == signature
            )
            if not candidates:
                return None
            slots.append((split, left_index, candidates))
    slots.sort(key=lambda slot: (len(slot[2]), slot[0], slot[1]))

    used = {"train": set(), "test": set()}
    assignment = {
        "train": [-1] * len(transformed["train"]),
        "test": [-1] * len(transformed["test"]),
    }

    def visit(
        position: int,
        mapping: dict[int, int],
        inverse: dict[int, int],
    ) -> tuple[dict[int, int], dict[str, list[int]]] | None:
        budget.tick()
        if position == len(slots):
            return mapping, assignment
        split, left_index, candidates = slots[position]
        for target_index in candidates:
            if target_index in used[split]:
                continue
            extended = _extend_color_bijection(
                transformed[split][left_index],
                target[split][target_index],
                mapping,
                inverse,
            )
            if extended is None:
                continue
            used[split].add(target_index)
            assignment[split][left_index] = target_index
            found = visit(position + 1, *extended)
            if found is not None:
                return found
            assignment[split][left_index] = -1
            used[split].remove(target_index)
        return None

    found = visit(0, {}, {})
    if found is None:
        return None
    mapping, final_assignment = found
    return IsomorphismWitness(
        d4_transform=transform_name,
        color_mapping=mapping,
        train_permutation=tuple(final_assignment["train"]),
        test_permutation=tuple(final_assignment["test"]),
    )


def verify_isomorphism_witness(
    left: dict[str, Any],
    right: dict[str, Any],
    witness: IsomorphismWitness,
) -> bool:
    """Replay a witness against every cell of complete task content."""

    if witness.d4_transform not in D4:
        return False
    mapping = witness.color_mapping
    if len(set(mapping.values())) != len(mapping):
        return False
    permutations = {
        "train": witness.train_permutation,
        "test": witness.test_permutation,
    }
    for split in ("train", "test"):
        permutation = permutations[split]
        if sorted(permutation) != list(range(len(right[split]))):
            return False
        if len(permutation) != len(left[split]):
            return False
        for left_index, right_index in enumerate(permutation):
            transformed = _transform_example(
                left[split][left_index], witness.d4_transform
            )
            target = right[split][right_index]
            if transformed.keys() != target.keys():
                return False
            for field, grid in transformed.items():
                try:
                    mapped = [[mapping[cell] for cell in row] for row in grid]
                except KeyError:
                    return False
                if mapped != target[field]:
                    return False
    return True


def find_verified_isomorphism(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    max_search_states: int = DEFAULT_MAX_SEARCH_STATES,
) -> IsomorphismSearchResult:
    """Find a replayable global D4/color witness or conservatively decline.

    ``inconclusive`` means the state cap was reached.  Callers must never merge
    such a pair; this permits false negatives but prevents an unverified merge.
    """

    validate_task(left, source="left task", require_test_outputs=True)
    validate_task(right, source="right task", require_test_outputs=True)
    if task_candidate_signature(left) != task_candidate_signature(right):
        return IsomorphismSearchResult("not-isomorphic", None, 0)

    budget = _SearchBudget(max_search_states)
    try:
        for transform_name in D4:
            witness = _search_one_transform(left, right, transform_name, budget)
            if witness is not None:
                if not verify_isomorphism_witness(left, right, witness):
                    raise AssertionError("constructed witness did not replay")
                return IsomorphismSearchResult("matched", witness, budget.states)
    except _SearchLimitReached:
        return IsomorphismSearchResult("inconclusive", None, budget.states)
    return IsomorphismSearchResult("not-isomorphic", None, budget.states)


def _load_training_source(
    name: str,
    path: str | Path,
    *,
    expected_count: int | None,
) -> TrainingSource:
    source = Path(path).resolve()
    if source.name != "training" or "evaluation" in {
        part.lower() for part in source.parts
    }:
        raise ValueError(
            f"development inputs must resolve to a directory named training: {source}"
        )
    if not source.is_dir():
        raise ValueError(f"training source is not a directory: {source}")
    paths = sorted(source.glob("*.json"))
    if expected_count is not None and len(paths) != expected_count:
        raise ValueError(
            f"{name}: expected {expected_count} task files, found {len(paths)}"
        )
    if not paths:
        raise ValueError(f"{name}: no JSON task files found")

    records: list[TaskRecord] = []
    tree_entries: list[dict[str, str]] = []
    seen_task_ids: set[str] = set()
    for task_path in paths:
        task_id = task_path.stem
        if task_id in seen_task_ids:
            raise ValueError(f"{name}: duplicate task id {task_id}")
        seen_task_ids.add(task_id)
        task = load_task(task_path, require_test_outputs=True)
        file_sha256 = sha256_file(task_path)
        relative_path = task_path.relative_to(source).as_posix()
        tree_entries.append(
            {"relative_path": relative_path, "file_sha256": file_sha256}
        )
        records.append(
            TaskRecord(
                record_id=f"{name}:{task_id}",
                source=name,
                task_id=task_id,
                relative_path=relative_path,
                source_file_sha256=file_sha256,
                full_task_content_sha256=fingerprint(_full_task_content(task)),
                task=task,
            )
        )
    return TrainingSource(
        name=name,
        path=source,
        records=tuple(records),
        tree_sha256=fingerprint(tree_entries),
    )


def cluster_task_records(
    records: Sequence[TaskRecord],
    *,
    max_search_states: int = DEFAULT_MAX_SEARCH_STATES,
) -> ClusterResult:
    ordered = sorted(records, key=lambda record: record.record_id)
    if len({record.record_id for record in ordered}) != len(ordered):
        raise ValueError("task record IDs must be unique")
    buckets: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(ordered):
        buckets[task_candidate_signature(record.task)].append(index)

    union_find = _UnionFind(len(ordered))
    verified_edges: list[dict[str, object]] = []
    inconclusive_pairs: list[dict[str, object]] = []
    candidate_pair_count = 0
    explored_search_states = 0
    for signature, indices in sorted(
        buckets.items(), key=lambda item: fingerprint(item[0])
    ):
        signature_sha256 = fingerprint(signature)
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                candidate_pair_count += 1
                left = ordered[left_index]
                right = ordered[right_index]
                search = find_verified_isomorphism(
                    left.task,
                    right.task,
                    max_search_states=max_search_states,
                )
                explored_search_states += search.explored_states
                if search.status == "inconclusive":
                    inconclusive_pairs.append(
                        {
                            "left_record_id": left.record_id,
                            "right_record_id": right.record_id,
                            "candidate_signature_sha256": signature_sha256,
                            "explored_states": search.explored_states,
                        }
                    )
                    continue
                if search.status != "matched":
                    continue
                if search.witness is None or not verify_isomorphism_witness(
                    left.task, right.task, search.witness
                ):
                    raise AssertionError("verified match is missing a replayable witness")
                relation = (
                    "exact-normalized-full-task"
                    if _full_task_content(left.task) == _full_task_content(right.task)
                    else "global-d4-color-isomorphic-full-task"
                )
                union_find.union(left_index, right_index)
                verified_edges.append(
                    {
                        "left_record_id": left.record_id,
                        "right_record_id": right.record_id,
                        "relation": relation,
                        "candidate_signature_sha256": signature_sha256,
                        "explored_states": search.explored_states,
                        "witness": search.witness.to_manifest(),
                    }
                )

    grouped: dict[int, list[TaskRecord]] = defaultdict(list)
    for index, record in enumerate(ordered):
        grouped[union_find.find(index)].append(record)
    groups = tuple(
        sorted(
            (tuple(sorted(group, key=lambda record: record.record_id)) for group in grouped.values()),
            key=lambda group: tuple(record.record_id for record in group),
        )
    )
    return ClusterResult(
        groups=groups,
        verified_edges=tuple(verified_edges),
        inconclusive_pairs=tuple(inconclusive_pairs),
        candidate_bucket_count=len(buckets),
        candidate_pair_count=candidate_pair_count,
        explored_search_states=explored_search_states,
    )


def _cluster_descriptor(group: Sequence[TaskRecord]) -> dict[str, object]:
    member_descriptors = [record.to_manifest() for record in group]
    member_digest = fingerprint(member_descriptors)
    representative = min(group, key=lambda record: record.record_id)
    source_counts = Counter(record.source for record in group)
    return {
        "cluster_id": member_digest,
        "cluster_id_scheme": (
            "sha256(canonical-json(sorted complete member audit records))"
        ),
        "member_count": len(group),
        "member_record_ids": [record.record_id for record in group],
        "member_source_counts": dict(sorted(source_counts.items())),
        "exact_content_group_count": len(
            {record.full_task_content_sha256 for record in group}
        ),
        "representative_record_id": representative.record_id,
    }


def _largest_remainder_quotas(
    count: int, split_weights: Sequence[tuple[str, int]]
) -> dict[str, int]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    if not split_weights or any(
        not name
        or isinstance(weight, bool)
        or not isinstance(weight, int)
        or weight <= 0
        for name, weight in split_weights
    ):
        raise ValueError("split weights must be named positive integers")
    if len({name for name, _ in split_weights}) != len(split_weights):
        raise ValueError("split names must be unique")
    total_weight = sum(weight for _, weight in split_weights)
    quotas = {
        name: count * weight // total_weight for name, weight in split_weights
    }
    remainder = count - sum(quotas.values())
    ranked = sorted(
        split_weights,
        key=lambda item: (-(count * item[1] % total_weight), item[0]),
    )
    for name, _ in ranked[:remainder]:
        quotas[name] += 1
    return quotas


def _audit_payload_digest(manifest: dict[str, object]) -> str:
    payload = deepcopy(manifest)
    digests = payload.get("digests")
    if not isinstance(digests, dict):
        raise ValueError("manifest digests must be an object")
    digests.pop("audit_payload_sha256", None)
    return fingerprint(payload)


def _build_contamination_annotation(
    ledger: Mapping[str, object] | None,
    *,
    records: Sequence[TaskRecord],
    clusters: Sequence[dict[str, object]],
    splits: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    """Annotate known ARC-1 exposure without changing any assignment input."""

    if ledger is None:
        return {
            "status": "overlap-ledger-not-provided",
            "annotation_only": True,
            "assignment_influence": False,
            "arc1_public_evaluation_claim": "blocked-pending-overlap-annotation",
            "arc1_clean_view": {
                "status": "required-not-materialized",
                "policy": (
                    "create and predeclare a separate overlap-excluded view before "
                    "any ARC-AGI-1-clean development claim"
                ),
            },
        }

    task_ids = ledger.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or any(not isinstance(task_id, str) for task_id in task_ids)
        or task_ids != sorted(task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError("ARC-1 overlap annotation requires a sorted unique ID ledger")
    if ledger.get("task_id_count") != len(task_ids):
        raise ValueError("ARC-1 overlap annotation count mismatch")
    if ledger.get("task_id_ledger_sha256") != fingerprint(task_ids):
        raise ValueError("ARC-1 overlap annotation digest mismatch")
    if ledger.get("direct_evaluation_files_opened") != 0:
        raise ValueError("overlap annotation must not open evaluation files")

    arc2_records = {
        record.task_id: record
        for record in records
        if record.source == "arc-agi-2-training"
    }
    missing_task_ids = sorted(set(task_ids) - set(arc2_records))
    if missing_task_ids:
        raise ValueError(
            "overlap ledger IDs are absent from ARC-AGI-2 training: "
            + ", ".join(missing_task_ids[:5])
        )
    flagged_record_ids = sorted(arc2_records[task_id].record_id for task_id in task_ids)
    cluster_by_record_id = {
        str(record_id): str(cluster["cluster_id"])
        for cluster in clusters
        for record_id in cluster["member_record_ids"]
    }
    cluster_by_id = {str(cluster["cluster_id"]): cluster for cluster in clusters}
    flagged_cluster_ids = sorted(
        {cluster_by_record_id[record_id] for record_id in flagged_record_ids}
    )
    flagged_cluster_member_record_ids = sorted(
        record_id
        for cluster_id in flagged_cluster_ids
        for record_id in cluster_by_id[cluster_id]["member_record_ids"]
    )
    flagged_cluster_set = set(flagged_cluster_ids)
    flagged_record_set = set(flagged_record_ids)
    by_split: dict[str, dict[str, int]] = {}
    for split_name, split in splits.items():
        split_clusters = set(split["cluster_ids"])
        split_records = set(split["task_record_ids"])
        excluded_clusters = split_clusters & flagged_cluster_set
        by_split[split_name] = {
            "flagged_cluster_count": len(excluded_clusters),
            "flagged_arc2_training_record_count": len(
                split_records & flagged_record_set
            ),
            "excluded_cluster_member_record_count": sum(
                int(cluster_by_id[cluster_id]["member_count"])
                for cluster_id in excluded_clusters
            ),
            "remaining_cluster_count_after_exclusion_without_reallocation": (
                len(split_clusters) - len(excluded_clusters)
            ),
        }

    reference = {
        key: value for key, value in ledger.items() if key != "task_ids"
    }
    return {
        "status": "annotated-contamination-aware-for-arc1",
        "annotation_only": True,
        "assignment_influence": False,
        "reference_audit": reference,
        "overlap_task_id_count": len(task_ids),
        "overlap_task_id_ledger_sha256": fingerprint(task_ids),
        "overlap_task_ids": task_ids,
        "flagged_arc2_training_record_count": len(flagged_record_ids),
        "flagged_arc2_training_record_ids": flagged_record_ids,
        "flagged_cluster_count": len(flagged_cluster_ids),
        "flagged_cluster_ids": flagged_cluster_ids,
        "flagged_cluster_member_record_count": len(
            flagged_cluster_member_record_ids
        ),
        "flagged_cluster_member_record_ids": flagged_cluster_member_record_ids,
        "by_split": by_split,
        "arc2_development_policy": (
            "the general draft may be used for ARC-AGI-2 development subject to "
            "the remaining protocol gates"
        ),
        "arc1_public_evaluation_claim": "contamination-aware-only",
        "arc1_clean_view": {
            "status": "required-not-materialized",
            "excluded_cluster_count_if_materialized": len(flagged_cluster_ids),
            "excluded_source_record_count_if_materialized": len(
                flagged_cluster_member_record_ids
            ),
            "remaining_cluster_count_if_materialized": (
                len(clusters) - len(flagged_cluster_ids)
            ),
            "remaining_source_record_count_if_materialized": (
                len(records) - len(flagged_cluster_member_record_ids)
            ),
            "policy": (
                "create and predeclare a separate overlap-excluded view before "
                "method selection for any ARC-AGI-1-clean configuration; do not "
                "reinterpret this general assignment as ARC-AGI-1-clean"
            ),
        },
    }


def build_development_manifest(
    source_directories: Mapping[str, str | Path],
    *,
    expected_counts: Mapping[str, int] | None = None,
    public_seed: int = DEFAULT_PUBLIC_SEED,
    split_weights: Sequence[tuple[str, int]] = DEFAULT_SPLIT_WEIGHTS,
    max_search_states: int = DEFAULT_MAX_SEARCH_STATES,
    arc1_overlap_ledger: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a byte-deterministic, training-only draft development manifest."""

    if public_seed != DEFAULT_PUBLIC_SEED:
        raise ValueError(
            f"the draft public seed is fixed at {DEFAULT_PUBLIC_SEED}"
        )
    if tuple(split_weights) != DEFAULT_SPLIT_WEIGHTS:
        raise ValueError(
            f"the draft split weights are fixed at {DEFAULT_SPLIT_WEIGHTS!r}"
        )
    if set(source_directories) != set(SOURCE_NAMES):
        raise ValueError(f"source directories must be exactly {SOURCE_NAMES!r}")
    expected = expected_counts or {}
    sources = [
        _load_training_source(
            name,
            source_directories[name],
            expected_count=expected.get(name),
        )
        for name in SOURCE_NAMES
    ]
    records = tuple(
        sorted(
            (record for source in sources for record in source.records),
            key=lambda record: record.record_id,
        )
    )
    clustering = cluster_task_records(
        records, max_search_states=max_search_states
    )
    clusters = [_cluster_descriptor(group) for group in clustering.groups]
    clusters.sort(key=lambda cluster: str(cluster["cluster_id"]))
    cluster_by_id = {str(cluster["cluster_id"]): cluster for cluster in clusters}
    if len(cluster_by_id) != len(clusters):
        raise AssertionError("cluster ID collision")
    record_by_id = {record.record_id: record for record in records}

    ranked_clusters = sorted(
        clusters,
        key=lambda cluster: (
            hashlib.sha256(
                f"{public_seed}\0{cluster['cluster_id']}".encode("utf-8")
            ).hexdigest(),
            str(cluster["cluster_id"]),
        ),
    )
    quotas = _largest_remainder_quotas(len(clusters), split_weights)
    assignments: dict[str, list[str]] = {}
    cursor = 0
    for split_name, _ in split_weights:
        next_cursor = cursor + quotas[split_name]
        assignments[split_name] = sorted(
            str(cluster["cluster_id"])
            for cluster in ranked_clusters[cursor:next_cursor]
        )
        cursor = next_cursor
    if cursor != len(clusters):
        raise AssertionError("cluster quota assignment did not consume all clusters")

    split_manifests: dict[str, dict[str, object]] = {}
    for split_name, _ in split_weights:
        cluster_ids = assignments[split_name]
        member_record_ids = sorted(
            record_id
            for cluster_id in cluster_ids
            for record_id in cluster_by_id[cluster_id]["member_record_ids"]
        )
        representatives = [
            {
                "cluster_id": cluster_id,
                "record_id": cluster_by_id[cluster_id]["representative_record_id"],
                "source": record_by_id[
                    str(cluster_by_id[cluster_id]["representative_record_id"])
                ].source,
                "task_id": record_by_id[
                    str(cluster_by_id[cluster_id]["representative_record_id"])
                ].task_id,
            }
            for cluster_id in cluster_ids
        ]
        source_records = Counter(
            record_by_id[record_id].source for record_id in member_record_ids
        )
        representative_sources = Counter(
            str(representative["source"]) for representative in representatives
        )
        task_ids_by_source = {
            source.name: sorted(
                record_by_id[record_id].task_id
                for record_id in member_record_ids
                if record_by_id[record_id].source == source.name
            )
            for source in sources
        }
        split_manifests[split_name] = {
            "cluster_count": len(cluster_ids),
            "deduplicated_task_count": len(cluster_ids),
            "source_record_count": len(member_record_ids),
            "source_record_counts": dict(sorted(source_records.items())),
            "representative_source_counts": dict(
                sorted(representative_sources.items())
            ),
            "cluster_ids": cluster_ids,
            "task_record_ids": member_record_ids,
            "task_ids_by_source": task_ids_by_source,
            "deduplicated_tasks": representatives,
        }

    contamination = _build_contamination_annotation(
        arc1_overlap_ledger,
        records=records,
        clusters=clusters,
        splits=split_manifests,
    )
    edge_counts = Counter(
        str(edge["relation"]) for edge in clustering.verified_edges
    )
    cross_source_cluster_count = sum(
        len(cluster["member_source_counts"]) > 1 for cluster in clusters
    )
    task_records_manifest = [record.to_manifest() for record in records]
    cluster_membership_payload = [
        {
            "cluster_id": cluster["cluster_id"],
            "member_record_ids": cluster["member_record_ids"],
        }
        for cluster in clusters
    ]
    assignment_payload = {
        name: split_manifests[name]["cluster_ids"] for name, _ in split_weights
    }
    verified_edges = list(clustering.verified_edges)
    inconclusive_pairs = list(clustering.inconclusive_pairs)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": "arc-agi-training-only-development-split-draft-20260806",
        "protocol_status": "draft-not-frozen",
        "scope": "ARC-AGI-1 training plus ARC-AGI-2 training only",
        "data_policy": {
            "permitted_sources": list(SOURCE_NAMES),
            "required_resolved_directory_name": "training",
            "evaluation_tasks_read": 0,
            "evaluation_labels_read": 0,
            "solution_files_read": 0,
            "prior_overlap_audit_id_ledgers_read": (
                1 if arc1_overlap_ledger is not None else 0
            ),
            "overlap_ledger_used_for_assignment": False,
            "training_test_outputs_included_in_full_content_clustering": True,
            "dev_audit_label_hiding": (
                "required downstream; this manifest contains IDs and hashes, not grids"
            ),
        },
        "sources": {source.name: source.to_manifest() for source in sources},
        "algorithm": {
            "name": "conservative-verified-global-d4-color-clustering-v1-draft",
            "merge_rule": (
                "union only after replaying one global D4 transform, one global "
                "bijection between all used colors, and complete train/test "
                "example permutations across every input/output cell"
            ),
            "full_content": (
                "all ARC training demonstration and test inputs and outputs; "
                "metadata names excluded; example order ignored within train/test"
            ),
            "candidate_filter": (
                "D4-invariant dimensions and per-grid color multiplicity signatures; "
                "filter equality alone never merges tasks"
            ),
            "d4_transforms": list(D4),
            "max_search_states_per_candidate_pair": max_search_states,
            "inconclusive_policy": "do not merge; false negatives are permitted",
            "false_merge_policy": "every merge edge must carry a replayable witness",
        },
        "allocation": {
            "public_seed": public_seed,
            "split_weights": {name: weight for name, weight in split_weights},
            "cluster_quotas": quotas,
            "cluster_rank": "sha256(utf8(decimal-seed + NUL + cluster-id))",
            "quota_method": (
                "integer largest remainder by cluster count; ties by split name"
            ),
            "representative_policy": (
                "lexicographically smallest source-qualified record ID in cluster"
            ),
            "assignment_inputs": (
                "training-derived cluster IDs, fixed public seed, and fixed weights "
                "only; contamination annotations are added after assignment"
            ),
        },
        "summary": {
            "source_record_count": len(records),
            "deduplicated_cluster_count": len(clusters),
            "deduplicated_record_count": len(records) - len(clusters),
            "multi_member_cluster_count": sum(
                int(cluster["member_count"]) > 1 for cluster in clusters
            ),
            "cross_source_cluster_count": cross_source_cluster_count,
            "largest_cluster_member_count": max(
                int(cluster["member_count"]) for cluster in clusters
            ),
            "candidate_bucket_count": clustering.candidate_bucket_count,
            "candidate_pair_count": clustering.candidate_pair_count,
            "verified_edge_count": len(verified_edges),
            "exact_verified_edge_count": edge_counts["exact-normalized-full-task"],
            "d4_color_nonexact_verified_edge_count": edge_counts[
                "global-d4-color-isomorphic-full-task"
            ],
            "inconclusive_pair_count": len(inconclusive_pairs),
            "explored_search_states": clustering.explored_search_states,
            "known_arc1_eval_overlap_task_id_count": contamination.get(
                "overlap_task_id_count", 0
            ),
            "known_arc1_eval_overlap_cluster_count": contamination.get(
                "flagged_cluster_count", 0
            ),
        },
        "digests": {
            "task_records_sha256": fingerprint(task_records_manifest),
            "cluster_membership_sha256": fingerprint(cluster_membership_payload),
            "verified_edges_sha256": fingerprint(verified_edges),
            "assignment_sha256": fingerprint(assignment_payload),
            "contamination_annotation_sha256": fingerprint(contamination),
        },
        "task_records": task_records_manifest,
        "clusters": clusters,
        "verified_edges": verified_edges,
        "inconclusive_pairs": inconclusive_pairs,
        "splits": split_manifests,
        "contamination": contamination,
        "limitations": [
            "This is a draft development manifest and is not protocol v1 or a protocol freeze.",
            "The verified relation covers global D4, global color bijection, and example order only; other semantic near-duplicates may remain separate.",
            "A search-cap hit is conservatively left unmerged, so the procedure may miss a valid isomorphism but cannot merge on an unverified candidate signature.",
            "Public training labels are intentionally used only to compare complete training-task content; no evaluation task or label is an input.",
            "The existing ARC-1-evaluation versus ARC-2-training audit contributes an ID ledger for annotation only; the generator never opens evaluation tasks and the ledger does not affect clustering or assignment.",
            "Because ARC-AGI-2 training contains known ARC-AGI-1 public-evaluation material, this general draft is contamination-aware for ARC-AGI-1 and cannot support an ARC-AGI-1-clean claim.",
            "A separate predeclared overlap-excluded ARC-AGI-1-clean development view has not been materialized and remains a protocol blocker.",
        ],
    }
    manifest["digests"]["audit_payload_sha256"] = _audit_payload_digest(manifest)
    validate_development_manifest(manifest)
    return manifest


def validate_development_manifest(manifest: dict[str, object]) -> None:
    """Validate manifest coverage, split isolation, counts, and all digests."""

    if manifest.get("protocol_status") != "draft-not-frozen":
        raise ValueError("development manifest must remain explicitly draft")
    data_policy = manifest.get("data_policy")
    if not isinstance(data_policy, dict) or any(
        data_policy.get(field) != 0
        for field in (
            "evaluation_tasks_read",
            "evaluation_labels_read",
            "solution_files_read",
        )
    ):
        raise ValueError("manifest does not declare a training-only input policy")
    if data_policy.get("overlap_ledger_used_for_assignment") is not False:
        raise ValueError("overlap annotation must not influence split assignment")
    allocation = manifest.get("allocation")
    if not isinstance(allocation, dict):
        raise ValueError("allocation must be an object")
    if allocation.get("public_seed") != DEFAULT_PUBLIC_SEED:
        raise ValueError("manifest does not use the fixed public seed")
    expected_weights = {name: weight for name, weight in DEFAULT_SPLIT_WEIGHTS}
    if allocation.get("split_weights") != expected_weights:
        raise ValueError("manifest does not use the fixed split weights")

    task_records = manifest.get("task_records")
    clusters = manifest.get("clusters")
    splits = manifest.get("splits")
    digests = manifest.get("digests")
    summary = manifest.get("summary")
    if not isinstance(task_records, list) or not isinstance(clusters, list):
        raise ValueError("task_records and clusters must be arrays")
    if not isinstance(splits, dict) or not isinstance(digests, dict):
        raise ValueError("splits and digests must be objects")
    if set(splits) != {name for name, _ in DEFAULT_SPLIT_WEIGHTS}:
        raise ValueError("manifest has unexpected development split names")
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")

    record_by_id: dict[str, dict[str, object]] = {}
    for record in task_records:
        if not isinstance(record, dict) or not isinstance(record.get("record_id"), str):
            raise ValueError("invalid task record")
        record_id = str(record["record_id"])
        if record_id in record_by_id:
            raise ValueError(f"duplicate task record ID: {record_id}")
        record_by_id[record_id] = record

    cluster_by_id: dict[str, dict[str, object]] = {}
    covered_records: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict) or not isinstance(
            cluster.get("cluster_id"), str
        ):
            raise ValueError("invalid cluster")
        cluster_id = str(cluster["cluster_id"])
        if cluster_id in cluster_by_id:
            raise ValueError(f"duplicate cluster ID: {cluster_id}")
        member_ids = cluster.get("member_record_ids")
        if not isinstance(member_ids, list) or not member_ids:
            raise ValueError(f"cluster has no members: {cluster_id}")
        if member_ids != sorted(member_ids) or len(member_ids) != len(set(member_ids)):
            raise ValueError(f"cluster members are not sorted and unique: {cluster_id}")
        if any(record_id not in record_by_id for record_id in member_ids):
            raise ValueError(f"cluster references an unknown task record: {cluster_id}")
        expected_id = fingerprint([record_by_id[record_id] for record_id in member_ids])
        if cluster_id != expected_id:
            raise ValueError(f"cluster digest mismatch: {cluster_id}")
        if cluster.get("member_count") != len(member_ids):
            raise ValueError(f"cluster member count mismatch: {cluster_id}")
        covered_records.extend(member_ids)
        cluster_by_id[cluster_id] = cluster
    if sorted(covered_records) != sorted(record_by_id):
        raise ValueError("clusters do not partition all source task records")
    if len(covered_records) != len(set(covered_records)):
        raise ValueError("a source task record appears in multiple clusters")

    assigned_clusters: list[str] = []
    for split_name, _ in DEFAULT_SPLIT_WEIGHTS:
        split = splits.get(split_name)
        if not isinstance(split, dict):
            raise ValueError(f"missing development split: {split_name}")
        cluster_ids = split.get("cluster_ids")
        if not isinstance(cluster_ids, list):
            raise ValueError(f"invalid cluster list for {split_name}")
        if cluster_ids != sorted(cluster_ids) or len(cluster_ids) != len(
            set(cluster_ids)
        ):
            raise ValueError(f"split clusters are not sorted and unique: {split_name}")
        if any(cluster_id not in cluster_by_id for cluster_id in cluster_ids):
            raise ValueError(f"split references unknown cluster: {split_name}")
        if split.get("cluster_count") != len(cluster_ids):
            raise ValueError(f"split cluster count mismatch: {split_name}")
        expected_records = sorted(
            record_id
            for cluster_id in cluster_ids
            for record_id in cluster_by_id[cluster_id]["member_record_ids"]
        )
        if split.get("task_record_ids") != expected_records:
            raise ValueError(f"split task record coverage mismatch: {split_name}")
        assigned_clusters.extend(cluster_ids)
    if sorted(assigned_clusters) != sorted(cluster_by_id):
        raise ValueError("development splits do not cover every cluster")
    if len(assigned_clusters) != len(set(assigned_clusters)):
        raise ValueError("a cluster crosses development splits")

    contamination = manifest.get("contamination")
    if not isinstance(contamination, dict):
        raise ValueError("contamination annotation must be an object")
    if contamination.get("assignment_influence") is not False:
        raise ValueError("contamination annotation influenced assignment")
    if contamination.get("status") == "annotated-contamination-aware-for-arc1":
        overlap_ids = contamination.get("overlap_task_ids")
        flagged_record_ids = contamination.get(
            "flagged_arc2_training_record_ids"
        )
        flagged_cluster_ids = contamination.get("flagged_cluster_ids")
        if not isinstance(overlap_ids, list) or not isinstance(
            flagged_record_ids, list
        ):
            raise ValueError("invalid ARC-1 overlap annotation ledgers")
        if not isinstance(flagged_cluster_ids, list):
            raise ValueError("invalid ARC-1 overlap cluster ledger")
        if overlap_ids != sorted(overlap_ids) or len(overlap_ids) != len(
            set(overlap_ids)
        ):
            raise ValueError("ARC-1 overlap IDs are not sorted and unique")
        if contamination.get("overlap_task_id_count") != len(overlap_ids):
            raise ValueError("ARC-1 overlap task count mismatch")
        if contamination.get("overlap_task_id_ledger_sha256") != fingerprint(
            overlap_ids
        ):
            raise ValueError("ARC-1 overlap task digest mismatch")
        expected_flagged_records = sorted(
            record_id
            for record_id, record in record_by_id.items()
            if record.get("source") == "arc-agi-2-training"
            and record.get("task_id") in set(overlap_ids)
        )
        if flagged_record_ids != expected_flagged_records:
            raise ValueError("ARC-1 overlap source-record annotation mismatch")
        expected_flagged_clusters = sorted(
            cluster_id
            for cluster_id, cluster in cluster_by_id.items()
            if set(cluster["member_record_ids"]) & set(flagged_record_ids)
        )
        if flagged_cluster_ids != expected_flagged_clusters:
            raise ValueError("ARC-1 overlap cluster annotation mismatch")
        if contamination.get("arc1_public_evaluation_claim") != (
            "contamination-aware-only"
        ):
            raise ValueError("ARC-1 claim boundary is not contamination-aware")
        arc1_clean_view = contamination.get("arc1_clean_view")
        if not isinstance(arc1_clean_view, dict) or arc1_clean_view.get(
            "status"
        ) != "required-not-materialized":
            raise ValueError("ARC-1-clean exclusion view blocker is missing")

    cluster_membership_payload = [
        {
            "cluster_id": cluster["cluster_id"],
            "member_record_ids": cluster["member_record_ids"],
        }
        for cluster in clusters
    ]
    assignment_payload = {
        name: splits[name]["cluster_ids"] for name, _ in DEFAULT_SPLIT_WEIGHTS
    }
    checks = {
        "task_records_sha256": fingerprint(task_records),
        "cluster_membership_sha256": fingerprint(cluster_membership_payload),
        "verified_edges_sha256": fingerprint(manifest.get("verified_edges", [])),
        "assignment_sha256": fingerprint(assignment_payload),
        "contamination_annotation_sha256": fingerprint(contamination),
        "audit_payload_sha256": _audit_payload_digest(manifest),
    }
    for name, observed in checks.items():
        if digests.get(name) != observed:
            raise ValueError(f"manifest digest mismatch: {name}")
    if summary.get("source_record_count") != len(task_records):
        raise ValueError("source record summary count mismatch")
    if summary.get("deduplicated_cluster_count") != len(clusters):
        raise ValueError("cluster summary count mismatch")


__all__ = [
    "DEFAULT_MAX_SEARCH_STATES",
    "DEFAULT_PUBLIC_SEED",
    "DEFAULT_SPLIT_WEIGHTS",
    "IsomorphismSearchResult",
    "IsomorphismWitness",
    "SOURCE_NAMES",
    "TaskRecord",
    "build_development_manifest",
    "canonical_json_bytes",
    "cluster_task_records",
    "find_verified_isomorphism",
    "fingerprint",
    "load_overlap_id_ledger",
    "task_candidate_signature",
    "validate_development_manifest",
    "verify_isomorphism_witness",
]
