# External baseline sources

Retrieved on 2026-08-04 as immutable GitHub source archives. Repository
metadata was read from `GET /repos/{owner}/{repo}` and each head SHA from
`GET /repos/{owner}/{repo}/commits/{default_branch}` in the GitHub REST API.
The shared environment's direct unauthenticated API quota returned HTTP 403,
so the same API responses were fetched through the read-only
`r.jina.ai/http://api.github.com/...` transport. The SHAs were cross-checked
against GitHub's branch commit pages and Atom feeds.

## Revisions

| Local path | Canonical repository | Default branch | Revision | Head commit time | Code license | Retained regular-file bytes |
| --- | --- | --- | --- | --- | --- | ---: |
| `CompressARC` | <https://github.com/iliao2345/CompressARC> | `master` | `83a22218024d46273eb32b769a906340202ffb4d` | 2026-01-05 17:56:27 UTC | MIT | 4,323,833 |
| `ARC-VSA-2025` | <https://github.com/ijoffe/ARC-VSA-2025> | `main` | `c031a9c6b4885ab03b28fbfdcd97b6b3693df564` | 2025-11-30 15:53:52 UTC | MIT | 178,159 |

The complete MIT texts and copyright notices are retained at
`CompressARC/LICENSE` and `ARC-VSA-2025/LICENSE`.

API endpoints used:

```text
https://api.github.com/repos/iliao2345/CompressARC
https://api.github.com/repos/iliao2345/CompressARC/commits/master
https://api.github.com/repos/ijoffe/ARC-VSA-2025
https://api.github.com/repos/ijoffe/ARC-VSA-2025/commits/main
```

## Archives and checksums

| Repository | Immutable codeload archive | Archive bytes | Archive SHA-256 | Retained-tree manifest SHA-256 |
| --- | --- | ---: | --- | --- |
| CompressARC | <https://codeload.github.com/iliao2345/CompressARC/tar.gz/83a22218024d46273eb32b769a906340202ffb4d> | 43,454,399 | `804fb32defaedfd48aad54aef08e0f9daf585fb27faedf81400c5c82db98c1b0` | `db41685bb9161aa2aa9727dea2a48a601285c04e821ca1b8e9e7d76251d202e7` |
| ARC-VSA-2025 | <https://codeload.github.com/ijoffe/ARC-VSA-2025/tar.gz/c031a9c6b4885ab03b28fbfdcd97b6b3693df564> | 34,648,215 | `20ba7ea57947e4ba74e6bb4d7cdecbee8ddde1755b8206d503cdcad016a18b20` | `c020a419166072ee2fb3e1bf292423fc4a699ab703fe5a34a861e2d30b815218` |

Archive SHA-256 values were calculated on the downloaded tarballs before
extraction. The tarballs were then removed rather than adding 78,102,614 bytes
of redundant compressed data to the workspace.

The retained-tree values hash a sorted manifest of each retained file's
SHA-256 and relative path. Reproduce them from `external/` with:

```bash
LC_ALL=C sha256sum CompressARC/*.py CompressARC/*.md CompressARC/LICENSE CompressARC/requirements.txt CompressARC/dataset/*.json | LC_ALL=C sort -k2 | sha256sum
LC_ALL=C sha256sum ARC-VSA-2025/.gitignore ARC-VSA-2025/*.md ARC-VSA-2025/LICENSE ARC-VSA-2025/requirements.txt ARC-VSA-2025/src/*.py | LC_ALL=C sort -k2 | sha256sum
```

## Storage selection

CompressARC retains all root Python files, `README.md`, `requirements.txt`,
`LICENSE`, and the six JSON files in `dataset/`. Its
`results_for_the_blog_post/` directory was excluded: 2,302 generated files and
103,340,246 uncompressed bytes, including two 38.4 MB prediction archives,
learned representations, and plots. The 193,911-byte README banner was also
excluded. The retained training and evaluation aggregates were checked against
the canonical task files in `third_party/arc-agi-1`; all 400 tasks in each
split match after the expected task-per-file to Kaggle-aggregate conversion.

ARC-VSA-2025 retains `.gitignore`, `README.md`, `requirements.txt`, `LICENSE`,
and every file under `src/`. Its `paper/` directory was excluded: 11 publication
artifacts and 38,458,836 uncompressed bytes, mostly `paper.pdf` and figures.

No checkpoint, model-weight, generated-result, or Git LFS pointer file is
retained. The filtered snapshots otherwise preserve upstream bytes and paths;
upstream source was not edited.

## Data licensing note

CompressARC's root MIT license covers its software but does not separately
identify the license of the bundled competition JSON. Its retained training and
evaluation content matches ARC-AGI-1, whose canonical source is Apache-2.0 and
is recorded in `third_party/SOURCES.md`. The test challenge and sample
submission were not independently matched to that canonical snapshot; treat
them under the applicable ARC Prize/Kaggle data terms rather than assuming the
software's MIT license applies.

See `docs/FIRST_RUN_PLAN.md` for environment, data-path, compatibility, and
smoke-test details.

## Phase-1 candidate snapshots (2026-08-04)

The following three repositories were acquired later on the same date for the
next Phase-1 preflight. GitHub repository metadata, commit metadata, and
`git/ref/heads/main` all agreed on each default branch and revision. Downloads
used immutable SHA-addressed GitHub codeload archives.

### Revisions

| Local path | Canonical repository | Default branch | Revision | Head commit time | Code license | Files | Retained regular-file bytes |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| `ARC_NCA` | <https://github.com/etimush/ARC_NCA> | `main` | `25d522bc766f9ddaebbf7dad63f58790fe7aa884` | 2025-05-09 14:26:05 UTC | Apache-2.0 | 11 | 132,673 |
| `GridCoder2024` | <https://github.com/SimonOuellette35/GridCoder2024> | `main` | `bf6136e5f57029dcbbb85242b8ffd8a1a241bb5f` | 2025-02-13 21:12:55 UTC | No license file; `NOASSERTION` | 29 | 1,577,677 |
| `ARC-AGI-Challenge-2024` (2D nGPT) | <https://github.com/jfpuget/ARC-AGI-Challenge-2024> | `main` | `e5420b10b9470b3b5c6548572768d2d4c15130f6` | 2024-11-22 16:42:40 UTC | Apache-2.0 | 10 | 1,091,417 |

The complete Apache-2.0 text is retained in both licensed snapshots as
`LICENSE`; the two files are identical and have SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
GitHub reports no detected license for GridCoder2024 and its source archive has
no license or notice file. Public visibility does not grant reuse rights; keep
that snapshot limited to audit until the author supplies licensing terms.

API endpoints used:

```text
https://api.github.com/repos/etimush/ARC_NCA
https://api.github.com/repos/etimush/ARC_NCA/commits/main
https://api.github.com/repos/etimush/ARC_NCA/git/ref/heads/main
https://api.github.com/repos/SimonOuellette35/GridCoder2024
https://api.github.com/repos/SimonOuellette35/GridCoder2024/commits/main
https://api.github.com/repos/SimonOuellette35/GridCoder2024/git/ref/heads/main
https://api.github.com/repos/jfpuget/ARC-AGI-Challenge-2024
https://api.github.com/repos/jfpuget/ARC-AGI-Challenge-2024/commits/main
https://api.github.com/repos/jfpuget/ARC-AGI-Challenge-2024/git/ref/heads/main
```

### Archives and checksums

| Repository | Immutable codeload archive | Archive bytes | Archive SHA-256 | Retained-tree manifest SHA-256 |
| --- | --- | ---: | --- | --- |
| ARC_NCA | <https://codeload.github.com/etimush/ARC_NCA/tar.gz/25d522bc766f9ddaebbf7dad63f58790fe7aa884> | 53,030 | `fcf71ce5eb16cf7093c48c7fafd2a4aec527613d663fc2af740a49e726fb3f28` | `fed00d51ae74e25a9ad5e794804ad4f31e003204d4386759eaa3758b3f1f53b1` |
| GridCoder2024 | <https://codeload.github.com/SimonOuellette35/GridCoder2024/tar.gz/bf6136e5f57029dcbbb85242b8ffd8a1a241bb5f> | 185,229 | `2ec267fd0f17e1ab20dd75c75cb4d7fc0d4e785ac52fd8212958276e58682051` | `a4d16a2f70c4bfe0573c4ef6e089786a0b0d44c6c6eb172c072eabeee19d153b` |
| 2D nGPT | <https://codeload.github.com/jfpuget/ARC-AGI-Challenge-2024/tar.gz/e5420b10b9470b3b5c6548572768d2d4c15130f6> | 834,787 | `4f06781d319e0ef061097f3b0fd037a2628f68c8e08e75c0c00fd2e20d73f7b0` | `f7d595edbc89619d83f1570532ff5ed58f155accac1dd53a6d11ea268a04d1dd` |

The retained-tree values hash a sorted manifest containing each retained
file's SHA-256, two spaces, its path relative to `external/`, and a newline.
Reproduce them from `external/` with:

```bash
LC_ALL=C sha256sum ARC_NCA/* | LC_ALL=C sort -k2 | sha256sum
LC_ALL=C sha256sum GridCoder2024/*.py GridCoder2024/*.md GridCoder2024/datasets/*.py GridCoder2024/datasets/generators/*.py GridCoder2024/model/*.py GridCoder2024/search/*.py GridCoder2024/utils/*.py | LC_ALL=C sort -k2 | sha256sum
LC_ALL=C sha256sum ARC-AGI-Challenge-2024/LICENSE ARC-AGI-Challenge-2024/README.md ARC-AGI-Challenge-2024/cfg/* ARC-AGI-Challenge-2024/code/* ARC-AGI-Challenge-2024/notebooks/* | LC_ALL=C sort -k2 | sha256sum
```

The three archives totaled 1,073,046 bytes and were removed after extraction,
byte comparison, and checksum verification.

### Storage selection

ARC_NCA retains its complete 132,673-byte archive: source modules, five
notebooks, README, and Apache license. It contains no checkpoint, video, paper,
or Git LFS pointer. The notebooks' small embedded outputs were retained to
preserve upstream bytes and execution context.

GridCoder2024 retains its complete 1,577,677-byte archive: all 28 source/README
files plus the empty `datasets/__init__.py`. The archive contains no model,
generated CSV, paper, license, or Git LFS pointer. The required external
`model_full.pth` and `ARC_gym` dependency were not downloaded.

The 2D nGPT archive contains 1,442,082 regular-file bytes. The filtered tree
retains README, Apache license, both source files, both configurations, and all
three workflow notebooks. It excludes 350,665 bytes: the 350,561-byte
`arc.pdf`, and 104 bytes in two placeholder readmes under malformed upstream
directory names `checkpoints /` and `input /` (each has a trailing space).
Those placeholders contain no fixture or configuration beyond statements that
checkpoints and competition data belong there. No model checkpoint, generated
re-ARC data, paper, archive, or Git LFS pointer is retained.

These additions retain 2,801,767 regular-file bytes in 50 files and omit
350,665 uncompressed bytes. Upstream source files were not edited. See
`docs/NEXT_RUN_PLAN.md` for entry points, external artifact sizes, data reuse,
compatibility risks, and smoke commands.
