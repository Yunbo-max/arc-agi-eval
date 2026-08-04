#!/usr/bin/env python3
"""Synchronize paper metadata and approved evidence with private Hub repos."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def portable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: portable(item)
            for key, item in value.items()
            if key not in {"path", "status_path"}
        }
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value


def push_metadata(api: HfApi, settings: dict[str, Any], papers: list[str]) -> None:
    for repo_id, repo_type, card in [
        (settings["dataset_repo"], "dataset", ROOT / "hub" / "data" / "README.md"),
        (settings["model_repo"], "model", ROOT / "hub" / "models" / "README.md"),
    ]:
        api.upload_file(
            path_or_fileobj=card,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=repo_type,
        )
    for name in ("baselines.json", "source_locks.json", "paper_assets.json"):
        api.upload_file(
            path_or_fileobj=ROOT / "configs" / name,
            path_in_repo=f"manifests/{name}",
            repo_id=settings["dataset_repo"],
            repo_type="dataset",
        )
    for paper in papers:
        readme = ROOT / "papers" / paper / "README.md"
        for repo_id, repo_type in [
            (settings["dataset_repo"], "dataset"),
            (settings["model_repo"], "model"),
        ]:
            api.upload_file(
                path_or_fileobj=readme,
                path_in_repo=f"papers/{paper}/README.md",
                repo_id=repo_id,
                repo_type=repo_type,
            )
        lock_document = load("source_locks.json")
        asset_root = Path(
            os.environ.get(
                lock_document["asset_root_env"], lock_document["default_asset_root"]
            )
        ).expanduser()
        status = asset_root / "status" / f"{paper}.json"
        if status.is_file():
            payload = json.dumps(
                portable(json.loads(status.read_text(encoding="utf-8"))),
                indent=2,
                sort_keys=True,
            ).encode("utf-8") + b"\n"
            api.upload_file(
                path_or_fileobj=io.BytesIO(payload),
                path_in_repo=f"papers/{paper}/preparation_status.json",
                repo_id=settings["dataset_repo"],
                repo_type="dataset",
            )


def main() -> int:
    assets = load("paper_assets.json")
    papers = sorted(assets["papers"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("push-metadata", "push", "pull"))
    parser.add_argument("--paper", action="append", choices=papers)
    args = parser.parse_args()
    selected = papers if not args.paper else list(dict.fromkeys(args.paper))
    settings = assets["storage"]
    api = HfApi()
    api.create_repo(settings["dataset_repo"], repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(settings["model_repo"], repo_type="model", private=True, exist_ok=True)
    if args.action in {"push-metadata", "push"}:
        push_metadata(api, settings, selected)
    if args.action == "push":
        for paper in selected:
            report = ROOT / "reports" / paper
            if report.is_dir():
                api.upload_folder(
                    folder_path=report,
                    path_in_repo=f"papers/{paper}/reports",
                    repo_id=settings["model_repo"],
                    repo_type="model",
                    ignore_patterns=["*.tmp", "__pycache__/**"],
                )
        for path in sorted((ROOT / "results").glob("*")):
            if path.is_file():
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"shared-results/{path.name}",
                    repo_id=settings["model_repo"],
                    repo_type="model",
                )
    if args.action == "pull":
        target = ROOT / ".assets" / "hub"
        for repo_id, repo_type, folder in [
            (settings["dataset_repo"], "dataset", "data"),
            (settings["model_repo"], "model", "models"),
        ]:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                allow_patterns=[f"papers/{paper}/**" for paper in selected],
                local_dir=target / folder,
            )
    print(f"{args.action}: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
