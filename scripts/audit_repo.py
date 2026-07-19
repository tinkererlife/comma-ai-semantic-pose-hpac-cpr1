#!/usr/bin/env python3
"""Fail closed on artifact drift, accidental bloat, duplicates, or secrets."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_REPOSITORY_BYTES = 5 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024
IGNORED_PARTS = {".git", ".pytest_cache", "__pycache__", "build", "work"}
SECRET_PATTERNS = {
    "GitHub token": re.compile(("gh" + "p_")[0:] + r"[A-Za-z0-9]{20,}"),
    "GitHub fine-grained token": re.compile(
        ("github" + "_pat_")[0:] + r"[A-Za-z0-9_]{20,}"
    ),
    "AWS access key": re.compile(("AK" + "IA")[0:] + r"[A-Z0-9]{16}"),
    "OpenAI-style key": re.compile(("sk" + "-")[0:] + r"[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(("BEGIN " + "PRIVATE KEY")[0:]),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def included_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & IGNORED_PARTS)
    )


def main() -> None:
    failures: list[str] = []
    files = included_files()
    total = sum(path.stat().st_size for path in files)
    if total > MAX_REPOSITORY_BYTES:
        failures.append(f"repository is too large: {total} > {MAX_REPOSITORY_BYTES}")

    for path in files:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            failures.append(
                f"file exceeds the 1 MiB cap: {path.relative_to(ROOT)} ({size})"
            )

    artifact_lock = json.loads((ROOT / "recipe/artifacts.json").read_text())
    for item in artifact_lock["artifacts"]:
        path = ROOT / item["path"]
        if not path.is_file():
            failures.append(f"missing artifact: {item['path']}")
            continue
        size = path.stat().st_size
        actual = digest(path)
        if size != item["bytes"] or actual != item["sha256"]:
            failures.append(
                f"artifact drift: {item['path']} "
                f"got {size} {actual}, expected {item['bytes']} {item['sha256']}"
            )

    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for path in files:
        if path.stat().st_size:
            groups[(path.stat().st_size, digest(path))].append(
                str(path.relative_to(ROOT))
            )
    for (_, _), names in groups.items():
        if len(names) > 1:
            failures.append("duplicate file content: " + ", ".join(names))

    for path in files:
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        text = raw.decode("utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label} pattern in {path.relative_to(ROOT)}")

    if failures:
        print("\n".join(f"ERROR: {message}" for message in failures), file=sys.stderr)
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "status": "passed",
                "files": len(files),
                "repository_bytes": total,
                "artifact_count": len(artifact_lock["artifacts"]),
                "duplicate_groups": 0,
                "secret_patterns": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
