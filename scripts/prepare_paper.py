#!/usr/bin/env python3
"""Prepare locked source, shared data, and approved public assets per paper."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def command(arguments: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(arguments)}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def prepare_source(
    paper_id: str,
    lock: dict[str, Any] | None,
    asset_root: Path,
) -> dict[str, Any]:
    if lock is None:
        return {"status": "blocked", "reason": "no verified public repository"}
    if "repository_path" in lock:
        path = (ROOT / lock["repository_path"]).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return {
            "status": "ready",
            "path": str(path),
            "revision": lock["revision"],
            "verification": "tracked retained source snapshot",
        }
    path = (asset_root / lock["asset_subpath"]).resolve()
    environment = dict(os.environ)
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", "--filter=blob:none", "--no-checkout", lock["url"], str(path)], env=environment)
        command(["git", "fetch", "--depth", "1", "origin", lock["revision"]], cwd=path, env=environment)
        command(["git", "checkout", "--detach", lock["revision"]], cwd=path, env=environment)
    observed = command(["git", "rev-parse", "HEAD"], cwd=path)
    if observed != lock["revision"]:
        command(["git", "fetch", "--depth", "1", "origin", lock["revision"]], cwd=path, env=environment)
        command(["git", "checkout", "--detach", lock["revision"]], cwd=path, env=environment)
        observed = command(["git", "rev-parse", "HEAD"], cwd=path)
    if observed != lock["revision"]:
        raise RuntimeError(f"{paper_id}: expected {lock['revision']}, found {observed}")
    return {"status": "ready", "path": str(path), "revision": observed}


def validate_shared_data(document: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in document["shared_data"].items():
        base = ROOT / spec["path"]
        training = len(list((base / "training").glob("*.json")))
        evaluation = len(list((base / "evaluation").glob("*.json")))
        if training != spec["training_tasks"] or evaluation != spec["evaluation_tasks"]:
            raise RuntimeError(
                f"{name}: expected {spec['training_tasks']}/{spec['evaluation_tasks']} tasks, "
                f"found {training}/{evaluation}"
            )
        result[name] = {
            "status": "ready",
            "path": str(base.resolve()),
            "revision": spec["revision"],
            "training_tasks": training,
            "evaluation_tasks": evaluation,
        }
    return result


def download_asset(
    key: str,
    spec: dict[str, Any],
    minimum_free_bytes: int,
) -> dict[str, Any]:
    free = shutil.disk_usage(ROOT).free
    expected = int(spec.get("size_bytes", 0))
    if free - expected < minimum_free_bytes:
        return {
            "status": "blocked",
            "reason": "free-space reserve",
            "free_bytes": free,
            "expected_bytes": expected,
        }
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset" if spec["kind"] == "dataset" else None,
            revision=spec["revision"],
            allow_patterns=spec.get("allow_patterns"),
        )
        return {
            "status": "ready",
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "path": str(Path(path).resolve()),
        }
    except Exception as error:
        return {
            "status": "blocked",
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "reason": f"{type(error).__name__}: {error}",
        }


def main() -> int:
    baseline = load("baselines.json")
    entries = {entry["id"]: entry for entry in baseline["entries"]}
    lock_document = load("source_locks.json")
    locks = lock_document["sources"]
    assets = load("paper_assets.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="append", choices=sorted(entries))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--download-public-assets", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    if not args.all and not args.paper:
        parser.error("select --all or at least one --paper")
    papers = sorted(entries) if args.all else list(dict.fromkeys(args.paper))
    asset_root = Path(
        os.environ.get(
            lock_document["asset_root_env"], lock_document["default_asset_root"]
        )
    ).expanduser().resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    shared_data = validate_shared_data(assets)
    minimum_free_bytes = int(args.minimum_free_gib * 1024**3)
    downloaded: dict[str, Any] = {}
    summaries = []
    for paper_id in papers:
        plan = assets["papers"][paper_id]
        source = prepare_source(paper_id, locks.get(paper_id), asset_root)
        paper_assets: dict[str, Any] = {}
        if args.download_public_assets:
            for key in plan["download_assets"]:
                if key not in downloaded:
                    downloaded[key] = download_asset(
                        key, assets["downloadable_assets"][key], minimum_free_bytes
                    )
                paper_assets[key] = downloaded[key]
        value = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "paper_id": paper_id,
            "source": source,
            "shared_data": shared_data,
            "assets": paper_assets,
            "ready_definition": plan["ready_definition"],
        }
        status_path = asset_root / "status" / f"{paper_id}.json"
        atomic_json(status_path, value)
        summaries.append(
            {
                "paper_id": paper_id,
                "source": source["status"],
                "assets": {key: item["status"] for key, item in paper_assets.items()},
                "status_path": str(status_path),
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
