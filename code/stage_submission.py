#!/usr/bin/env python3
"""Stage the minimal runnable CPR1 submission directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


RUNTIME_FILES = (
    "inflate.sh",
    "inflate.py",
    "carrier_codec.py",
    "hpac_integer.py",
    "hpac_integer_sparse.py",
    "integer_model_io.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p":
            raise ValueError("CPR1 archive must contain exactly one member named p")
        if infos[0].compress_type != zipfile.ZIP_STORED:
            raise ValueError("CPR1 archive member p must be stored")

    args.submission_dir.mkdir(parents=True, exist_ok=True)
    staged = {}
    sources = {"archive.zip": args.archive}
    sources.update({
        name: args.code_root / name
        for name in RUNTIME_FILES
    })
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = args.submission_dir / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        if name.endswith(".sh"):
            destination.chmod(destination.stat().st_mode | 0o111)
        staged[name] = {
            "bytes": destination.stat().st_size,
            "sha256": digest(destination),
        }

    result = {
        "schema_version": 1,
        "submission_dir": str(args.submission_dir.resolve()),
        "files": staged,
        "charged_file": "archive.zip",
        "charged_bytes": staged["archive.zip"]["bytes"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
