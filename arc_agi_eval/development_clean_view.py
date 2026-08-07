"""Materialize a draft ARC-1 known-overlap-excluded development view.

The view is a pure filter over a locked training-only development manifest.  It
never opens ARC task files, never reads evaluation data, and never reallocates
clusters after exclusion.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .development_split import (
    DEFAULT_SPLIT_WEIGHTS,
    fingerprint,
    sha256_file,
    validate_development_manifest,
)


SPLIT_NAMES = tuple(name for name, _ in DEFAULT_SPLIT_WEIGHTS)


def load_locked_source_manifest(
    path: str | Path, *, expected_sha256: str
) -> dict[str, object]:
    """Load and validate one immutable source manifest without following paths."""

    source = Path(path).resolve()
    observed_sha256 = sha256_file(source)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "source development manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, found {observed_sha256}"
        )
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source development manifest must be a JSON object")
    validate_development_manifest(value)
    return value


def _audit_payload_digest(view: dict[str, object]) -> str:
    payload = deepcopy(view)
    digests = payload.get("digests")
    if not isinstance(digests, dict):
        raise ValueError("view digests must be an object")
    digests.pop("audit_payload_sha256", None)
    return fingerprint(payload)


def _record_counts(
    record_ids: list[str], record_by_id: dict[str, dict[str, object]]
) -> dict[str, int]:
    return dict(
        sorted(Counter(str(record_by_id[record_id]["source"]) for record_id in record_ids).items())
    )


def _task_ids_by_source(
    record_ids: list[str],
    record_by_id: dict[str, dict[str, object]],
    source_names: list[str],
) -> dict[str, list[str]]:
    return {
        source: sorted(
            str(record_by_id[record_id]["task_id"])
            for record_id in record_ids
            if record_by_id[record_id]["source"] == source
        )
        for source in source_names
    }


def _cluster_membership_payload(
    clusters: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "cluster_id": cluster["cluster_id"],
            "member_record_ids": cluster["member_record_ids"],
        }
        for cluster in clusters
    ]


def build_arc1_clean_development_view(
    source_manifest: dict[str, object],
    *,
    source_manifest_reference: str,
    source_manifest_file_sha256: str,
) -> dict[str, object]:
    """Filter all annotated ARC-1 overlap clusters without reallocation."""

    validate_development_manifest(source_manifest)
    if source_manifest.get("protocol_status") != "draft-not-frozen":
        raise ValueError("source development manifest must remain draft")
    contamination = source_manifest.get("contamination")
    if not isinstance(contamination, dict) or contamination.get("status") != (
        "annotated-contamination-aware-for-arc1"
    ):
        raise ValueError("source manifest lacks the authoritative ARC-1 annotation")
    flagged_cluster_ids = contamination.get("flagged_cluster_ids")
    if not isinstance(flagged_cluster_ids, list) or flagged_cluster_ids != sorted(
        flagged_cluster_ids
    ):
        raise ValueError("flagged cluster ledger must be a sorted array")
    if len(flagged_cluster_ids) != len(set(flagged_cluster_ids)):
        raise ValueError("flagged cluster ledger contains duplicates")
    if contamination.get("flagged_cluster_count") != len(flagged_cluster_ids):
        raise ValueError("flagged cluster count does not match its ledger")

    task_records = source_manifest.get("task_records")
    clusters = source_manifest.get("clusters")
    source_splits = source_manifest.get("splits")
    sources = source_manifest.get("sources")
    if not isinstance(task_records, list) or not isinstance(clusters, list):
        raise ValueError("source task records and clusters must be arrays")
    if not isinstance(source_splits, dict) or not isinstance(sources, dict):
        raise ValueError("source splits and sources must be objects")
    record_by_id = {str(record["record_id"]): record for record in task_records}
    cluster_by_id = {str(cluster["cluster_id"]): cluster for cluster in clusters}
    flagged_cluster_set = set(flagged_cluster_ids)
    if not flagged_cluster_set.issubset(cluster_by_id):
        raise ValueError("flagged cluster ledger references an unknown cluster")
    excluded_record_ids = sorted(
        str(record_id)
        for cluster_id in flagged_cluster_ids
        for record_id in cluster_by_id[cluster_id]["member_record_ids"]
    )
    annotated_excluded_records = contamination.get(
        "flagged_cluster_member_record_ids"
    )
    if annotated_excluded_records != excluded_record_ids:
        raise ValueError("flagged record ledger does not match flagged clusters")
    excluded_record_set = set(excluded_record_ids)

    retained_task_records = [
        record
        for record in task_records
        if str(record["record_id"]) not in excluded_record_set
    ]
    retained_clusters = [
        cluster
        for cluster in clusters
        if str(cluster["cluster_id"]) not in flagged_cluster_set
    ]
    retained_record_ids = [
        str(record["record_id"]) for record in retained_task_records
    ]
    retained_cluster_ids = [
        str(cluster["cluster_id"]) for cluster in retained_clusters
    ]
    source_names = sorted(str(source) for source in sources)

    source_edges = source_manifest.get("verified_edges", [])
    if not isinstance(source_edges, list):
        raise ValueError("source verified edges must be an array")
    retained_record_set = set(retained_record_ids)
    retained_edges = [
        edge
        for edge in source_edges
        if edge["left_record_id"] in retained_record_set
        and edge["right_record_id"] in retained_record_set
    ]
    source_inconclusive = source_manifest.get("inconclusive_pairs", [])
    if not isinstance(source_inconclusive, list):
        raise ValueError("source inconclusive pairs must be an array")
    retained_inconclusive = [
        pair
        for pair in source_inconclusive
        if pair["left_record_id"] in retained_record_set
        and pair["right_record_id"] in retained_record_set
    ]

    split_views: dict[str, dict[str, object]] = {}
    for split_name in SPLIT_NAMES:
        source_split = source_splits[split_name]
        original_cluster_ids = list(source_split["cluster_ids"])
        original_record_ids = list(source_split["task_record_ids"])
        excluded_split_clusters = sorted(
            cluster_id
            for cluster_id in original_cluster_ids
            if cluster_id in flagged_cluster_set
        )
        excluded_split_records = sorted(
            record_id
            for cluster_id in excluded_split_clusters
            for record_id in cluster_by_id[cluster_id]["member_record_ids"]
        )
        remaining_cluster_ids = [
            cluster_id
            for cluster_id in original_cluster_ids
            if cluster_id not in flagged_cluster_set
        ]
        remaining_record_ids = sorted(
            record_id
            for cluster_id in remaining_cluster_ids
            for record_id in cluster_by_id[cluster_id]["member_record_ids"]
        )
        if remaining_record_ids != sorted(
            set(original_record_ids) - set(excluded_split_records)
        ):
            raise AssertionError("split record filtering is not a pure exclusion")
        deduplicated_tasks = [
            task
            for task in source_split["deduplicated_tasks"]
            if task["cluster_id"] not in flagged_cluster_set
        ]
        task_ids_by_source = _task_ids_by_source(
            remaining_record_ids, record_by_id, source_names
        )
        split_digest_payload = {
            "cluster_ids": remaining_cluster_ids,
            "task_record_ids": remaining_record_ids,
            "task_ids_by_source": task_ids_by_source,
            "deduplicated_tasks": deduplicated_tasks,
        }
        split_views[split_name] = {
            "assignment_policy": "source assignment retained; no reallocation",
            "original_cluster_count": len(original_cluster_ids),
            "excluded_cluster_count": len(excluded_split_clusters),
            "cluster_count": len(remaining_cluster_ids),
            "original_source_record_count": len(original_record_ids),
            "excluded_source_record_count": len(excluded_split_records),
            "source_record_count": len(remaining_record_ids),
            "source_record_counts": _record_counts(
                remaining_record_ids, record_by_id
            ),
            "excluded_source_record_counts": _record_counts(
                excluded_split_records, record_by_id
            ),
            "representative_source_counts": dict(
                sorted(
                    Counter(str(task["source"]) for task in deduplicated_tasks).items()
                )
            ),
            "cluster_ids": remaining_cluster_ids,
            "task_record_ids": remaining_record_ids,
            "task_ids_by_source": task_ids_by_source,
            "deduplicated_tasks": deduplicated_tasks,
            "excluded_cluster_ids": excluded_split_clusters,
            "excluded_task_record_ids": excluded_split_records,
            "digests": {
                "cluster_ids_sha256": fingerprint(remaining_cluster_ids),
                "task_record_ids_sha256": fingerprint(remaining_record_ids),
                "task_ids_by_source_sha256": fingerprint(task_ids_by_source),
                "deduplicated_tasks_sha256": fingerprint(deduplicated_tasks),
                "split_payload_sha256": fingerprint(split_digest_payload),
            },
        }

    excluded_task_ids_by_source = _task_ids_by_source(
        excluded_record_ids, record_by_id, source_names
    )
    exclusion_payload = {
        "cluster_ids": flagged_cluster_ids,
        "task_record_ids": excluded_record_ids,
        "task_ids_by_source": excluded_task_ids_by_source,
    }
    retained_assignment = {
        name: split_views[name]["cluster_ids"] for name in SPLIT_NAMES
    }
    source_summary = source_manifest["summary"]
    view: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": (
            "arc-agi-1-known-overlap-excluded-development-view-draft-20260806"
        ),
        "protocol_status": "draft-not-frozen",
        "view_status": "materialized-known-overlap-excluded-draft",
        "scope": (
            "pure cluster exclusion from the locked training-only development "
            "manifest using its existing ARC-1 overlap annotation"
        ),
        "source_manifest": {
            "reference": source_manifest_reference,
            "file_sha256": source_manifest_file_sha256,
            "manifest_id": source_manifest["manifest_id"],
            "protocol_status": source_manifest["protocol_status"],
            "canonical_payload_sha256": source_manifest["digests"][
                "audit_payload_sha256"
            ],
            "cluster_membership_sha256": source_manifest["digests"][
                "cluster_membership_sha256"
            ],
            "assignment_sha256": source_manifest["digests"][
                "assignment_sha256"
            ],
            "contamination_annotation_sha256": source_manifest["digests"][
                "contamination_annotation_sha256"
            ],
        },
        "data_policy": {
            "source_manifest_files_read": 1,
            "training_task_files_read": 0,
            "evaluation_task_files_read": 0,
            "evaluation_label_files_read": 0,
            "solution_files_read": 0,
            "network_used": False,
            "cluster_reallocation_performed": False,
            "exclusion_source": (
                "source manifest contamination.flagged_cluster_ids only"
            ),
        },
        "sources": source_manifest["sources"],
        "exclusion": {
            "policy": "exclude complete flagged clusters; never individual members",
            "known_overlap_scope": (
                "376-ID ledger from the prior ARC-1 evaluation versus ARC-2 "
                "training overlap audit; no evaluation file is opened here"
            ),
            "source_annotation_sha256": source_manifest["digests"][
                "contamination_annotation_sha256"
            ],
            "known_overlap_task_id_count": source_manifest["contamination"][
                "overlap_task_id_count"
            ],
            "known_overlap_task_id_ledger_sha256": source_manifest[
                "contamination"
            ]["overlap_task_id_ledger_sha256"],
            "flagged_arc2_training_record_count": source_manifest[
                "contamination"
            ]["flagged_arc2_training_record_count"],
            "excluded_cluster_count": len(flagged_cluster_ids),
            "excluded_source_record_count": len(excluded_record_ids),
            "excluded_source_record_counts": _record_counts(
                excluded_record_ids, record_by_id
            ),
            "excluded_cluster_ids": flagged_cluster_ids,
            "excluded_task_record_ids": excluded_record_ids,
            "excluded_task_ids_by_source": excluded_task_ids_by_source,
        },
        "summary": {
            "source_cluster_count": source_summary["deduplicated_cluster_count"],
            "source_record_count": source_summary["source_record_count"],
            "excluded_cluster_count": len(flagged_cluster_ids),
            "excluded_source_record_count": len(excluded_record_ids),
            "remaining_cluster_count": len(retained_clusters),
            "remaining_source_record_count": len(retained_task_records),
            "remaining_source_record_counts": _record_counts(
                retained_record_ids, record_by_id
            ),
            "remaining_verified_edge_count": len(retained_edges),
            "remaining_inconclusive_pair_count": len(retained_inconclusive),
        },
        "digests": {
            "retained_task_records_sha256": fingerprint(retained_task_records),
            "retained_cluster_membership_sha256": fingerprint(
                _cluster_membership_payload(retained_clusters)
            ),
            "retained_verified_edges_sha256": fingerprint(retained_edges),
            "retained_assignment_sha256": fingerprint(retained_assignment),
            "exclusion_sha256": fingerprint(exclusion_payload),
        },
        "task_records": retained_task_records,
        "clusters": retained_clusters,
        "verified_edges": retained_edges,
        "inconclusive_pairs": retained_inconclusive,
        "splits": split_views,
        "limitations": [
            "This is a draft view and is not protocol v1 or a protocol freeze.",
            "Clean means only that every cluster flagged by the locked 376-ID overlap ledger was removed; it does not prove absence of renamed semantic overlap, researcher exposure, or model pretraining contamination.",
            "The source split assignments are retained without reallocation, so the remaining split proportions are intentionally no longer 70/15/15.",
            "dev-audit label hiding still requires a separate challenge-only runtime view and process firewall.",
        ],
        "claim_boundary": (
            "This materializes a known-overlap-excluded ARC-AGI-1 development "
            "draft from a locked training-only manifest. It is not a protocol "
            "freeze, historical decontamination proof, benchmark run, or score."
        ),
    }
    view["digests"]["audit_payload_sha256"] = _audit_payload_digest(view)
    validate_arc1_clean_development_view(view, source_manifest=source_manifest)
    return view


def validate_arc1_clean_development_view(
    view: dict[str, object],
    *,
    source_manifest: dict[str, object] | None = None,
    observed_source_manifest_file_sha256: str | None = None,
) -> None:
    """Validate exclusions, retained coverage, split isolation, and digests."""

    if view.get("protocol_status") != "draft-not-frozen":
        raise ValueError("ARC-1 clean view must remain explicitly draft")
    data_policy = view.get("data_policy")
    if not isinstance(data_policy, dict):
        raise ValueError("clean-view data policy must be an object")
    for field in (
        "training_task_files_read",
        "evaluation_task_files_read",
        "evaluation_label_files_read",
        "solution_files_read",
    ):
        if data_policy.get(field) != 0:
            raise ValueError(f"clean view violates zero-read policy: {field}")
    if data_policy.get("cluster_reallocation_performed") is not False:
        raise ValueError("clean view must not reallocate clusters")

    source_reference = view.get("source_manifest")
    exclusion = view.get("exclusion")
    task_records = view.get("task_records")
    clusters = view.get("clusters")
    splits = view.get("splits")
    digests = view.get("digests")
    summary = view.get("summary")
    if not isinstance(source_reference, dict) or not isinstance(exclusion, dict):
        raise ValueError("source reference and exclusion must be objects")
    if not isinstance(task_records, list) or not isinstance(clusters, list):
        raise ValueError("retained task records and clusters must be arrays")
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ValueError("clean view has unexpected split names")
    if not isinstance(digests, dict) or not isinstance(summary, dict):
        raise ValueError("clean-view digests and summary must be objects")

    excluded_cluster_ids = exclusion.get("excluded_cluster_ids")
    excluded_record_ids = exclusion.get("excluded_task_record_ids")
    if not isinstance(excluded_cluster_ids, list) or not isinstance(
        excluded_record_ids, list
    ):
        raise ValueError("excluded cluster and record ledgers must be arrays")
    if excluded_cluster_ids != sorted(excluded_cluster_ids) or len(
        excluded_cluster_ids
    ) != len(set(excluded_cluster_ids)):
        raise ValueError("excluded cluster ledger is not sorted and unique")
    if excluded_record_ids != sorted(excluded_record_ids) or len(
        excluded_record_ids
    ) != len(set(excluded_record_ids)):
        raise ValueError("excluded task-record ledger is not sorted and unique")
    excluded_cluster_set = set(excluded_cluster_ids)
    excluded_record_set = set(excluded_record_ids)

    record_by_id: dict[str, dict[str, object]] = {}
    for record in task_records:
        if not isinstance(record, dict) or not isinstance(record.get("record_id"), str):
            raise ValueError("invalid retained task record")
        record_id = str(record["record_id"])
        if record_id in record_by_id or record_id in excluded_record_set:
            raise ValueError("retained record is duplicate or flagged")
        record_by_id[record_id] = record
    cluster_by_id: dict[str, dict[str, object]] = {}
    covered_records: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict) or not isinstance(
            cluster.get("cluster_id"), str
        ):
            raise ValueError("invalid retained cluster")
        cluster_id = str(cluster["cluster_id"])
        if cluster_id in cluster_by_id or cluster_id in excluded_cluster_set:
            raise ValueError("retained cluster is duplicate or flagged")
        member_ids = cluster.get("member_record_ids")
        if not isinstance(member_ids, list) or not member_ids:
            raise ValueError("retained cluster has no members")
        if any(record_id not in record_by_id for record_id in member_ids):
            raise ValueError("retained cluster references missing or flagged records")
        expected_cluster_id = fingerprint(
            [record_by_id[record_id] for record_id in member_ids]
        )
        if cluster_id != expected_cluster_id:
            raise ValueError("retained cluster digest mismatch")
        covered_records.extend(member_ids)
        cluster_by_id[cluster_id] = cluster
    if sorted(covered_records) != sorted(record_by_id) or len(covered_records) != len(
        set(covered_records)
    ):
        raise ValueError("retained clusters do not partition retained records")

    assigned_clusters: list[str] = []
    for split_name in SPLIT_NAMES:
        split = splits[split_name]
        if not isinstance(split, dict):
            raise ValueError("clean-view split must be an object")
        cluster_ids = split.get("cluster_ids")
        task_record_ids = split.get("task_record_ids")
        if not isinstance(cluster_ids, list) or not isinstance(task_record_ids, list):
            raise ValueError("clean-view split ledgers must be arrays")
        if set(cluster_ids) & excluded_cluster_set:
            raise ValueError("a flagged cluster remains in a clean-view split")
        expected_records = sorted(
            record_id
            for cluster_id in cluster_ids
            for record_id in cluster_by_id[cluster_id]["member_record_ids"]
        )
        if task_record_ids != expected_records:
            raise ValueError("clean-view split record coverage mismatch")
        if split.get("cluster_count") != len(cluster_ids):
            raise ValueError("clean-view split cluster count mismatch")
        if split.get("source_record_count") != len(task_record_ids):
            raise ValueError("clean-view split source-record count mismatch")
        split_payload = {
            "cluster_ids": cluster_ids,
            "task_record_ids": task_record_ids,
            "task_ids_by_source": split["task_ids_by_source"],
            "deduplicated_tasks": split["deduplicated_tasks"],
        }
        expected_split_digests = {
            "cluster_ids_sha256": fingerprint(cluster_ids),
            "task_record_ids_sha256": fingerprint(task_record_ids),
            "task_ids_by_source_sha256": fingerprint(
                split["task_ids_by_source"]
            ),
            "deduplicated_tasks_sha256": fingerprint(
                split["deduplicated_tasks"]
            ),
            "split_payload_sha256": fingerprint(split_payload),
        }
        if split.get("digests") != expected_split_digests:
            raise ValueError("clean-view split digest mismatch")
        assigned_clusters.extend(cluster_ids)
    if sorted(assigned_clusters) != sorted(cluster_by_id) or len(
        assigned_clusters
    ) != len(set(assigned_clusters)):
        raise ValueError("clean-view clusters cross splits or lack coverage")

    assignment_payload = {
        name: splits[name]["cluster_ids"] for name in SPLIT_NAMES
    }
    exclusion_payload = {
        "cluster_ids": excluded_cluster_ids,
        "task_record_ids": excluded_record_ids,
        "task_ids_by_source": exclusion["excluded_task_ids_by_source"],
    }
    expected_digests = {
        "retained_task_records_sha256": fingerprint(task_records),
        "retained_cluster_membership_sha256": fingerprint(
            _cluster_membership_payload(clusters)
        ),
        "retained_verified_edges_sha256": fingerprint(
            view.get("verified_edges", [])
        ),
        "retained_assignment_sha256": fingerprint(assignment_payload),
        "exclusion_sha256": fingerprint(exclusion_payload),
        "audit_payload_sha256": _audit_payload_digest(view),
    }
    for name, observed in expected_digests.items():
        if digests.get(name) != observed:
            raise ValueError(f"clean-view digest mismatch: {name}")
    if summary.get("remaining_cluster_count") != len(clusters):
        raise ValueError("clean-view remaining-cluster summary mismatch")
    if summary.get("remaining_source_record_count") != len(task_records):
        raise ValueError("clean-view remaining-record summary mismatch")
    if exclusion.get("excluded_cluster_count") != len(excluded_cluster_ids):
        raise ValueError("clean-view excluded-cluster summary mismatch")
    if exclusion.get("excluded_source_record_count") != len(excluded_record_ids):
        raise ValueError("clean-view excluded-record summary mismatch")

    if source_manifest is None:
        return
    validate_development_manifest(source_manifest)
    if observed_source_manifest_file_sha256 is not None and source_reference.get(
        "file_sha256"
    ) != observed_source_manifest_file_sha256:
        raise ValueError("clean view references the wrong source-manifest file hash")
    source_clusters = {
        str(cluster["cluster_id"]): cluster for cluster in source_manifest["clusters"]
    }
    source_records = {
        str(record["record_id"]): record
        for record in source_manifest["task_records"]
    }
    expected_excluded_clusters = source_manifest["contamination"][
        "flagged_cluster_ids"
    ]
    if excluded_cluster_ids != expected_excluded_clusters:
        raise ValueError("clean view does not exclude the source flagged-cluster ledger")
    expected_excluded_records = sorted(
        record_id
        for cluster_id in expected_excluded_clusters
        for record_id in source_clusters[cluster_id]["member_record_ids"]
    )
    if excluded_record_ids != expected_excluded_records:
        raise ValueError("clean view does not exclude every flagged-cluster member")
    expected_retained_cluster_ids = sorted(
        set(source_clusters) - set(expected_excluded_clusters)
    )
    if sorted(cluster_by_id) != expected_retained_cluster_ids:
        raise ValueError("clean view retained-cluster coverage differs from source")
    expected_retained_record_ids = sorted(
        set(source_records) - set(expected_excluded_records)
    )
    if sorted(record_by_id) != expected_retained_record_ids:
        raise ValueError("clean view retained-record coverage differs from source")
    for cluster_id, cluster in cluster_by_id.items():
        if cluster != source_clusters[cluster_id]:
            raise ValueError("clean view modified a retained cluster")
    for record_id, record in record_by_id.items():
        if record != source_records[record_id]:
            raise ValueError("clean view modified a retained task record")
    for split_name in SPLIT_NAMES:
        expected_ids = [
            cluster_id
            for cluster_id in source_manifest["splits"][split_name]["cluster_ids"]
            if cluster_id not in excluded_cluster_set
        ]
        if splits[split_name]["cluster_ids"] != expected_ids:
            raise ValueError("clean view reallocated or reordered retained clusters")
    source_digests = source_manifest["digests"]
    source_digest_checks = {
        "canonical_payload_sha256": source_digests["audit_payload_sha256"],
        "cluster_membership_sha256": source_digests[
            "cluster_membership_sha256"
        ],
        "assignment_sha256": source_digests["assignment_sha256"],
        "contamination_annotation_sha256": source_digests[
            "contamination_annotation_sha256"
        ],
    }
    for name, expected in source_digest_checks.items():
        if source_reference.get(name) != expected:
            raise ValueError(f"clean view source-manifest digest mismatch: {name}")


__all__ = [
    "SPLIT_NAMES",
    "build_arc1_clean_development_view",
    "load_locked_source_manifest",
    "validate_arc1_clean_development_view",
]
