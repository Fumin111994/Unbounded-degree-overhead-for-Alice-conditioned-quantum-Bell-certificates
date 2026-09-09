#!/usr/bin/env python3
"""Verify the pristine companion dataset using the repository's SHA-256 manifest."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path, manifest: Path) -> list[str]:
    root = root.resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    failures = []
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative) or ":" in str(relative):
            failures.append(f"Invalid relative path: {relative}")
            continue
        target = root.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(root):
            failures.append(f"Path escapes data root: {relative}")
        elif not target.is_file():
            failures.append(f"Missing: {relative}")
        elif target.stat().st_size != entry["bytes"] or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"Changed: {relative}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory containing artifacts/ and figures/")
    args = parser.parse_args()
    manifest = ROOT / "DATA_MANIFEST.json"
    if not manifest.is_file():
        print("FAIL: repository DATA_MANIFEST.json is missing", file=sys.stderr)
        return 1
    failures = verify(args.root, manifest)
    if failures:
        print("FAIL: dataset integrity check")
        print("\n".join(failures))
        print("Extract the companion data ZIP into the repository root. Regenerated reports can differ from the archive.")
        return 1
    count = len(json.loads(manifest.read_text(encoding="utf-8"))["files"])
    print(f"PASS: {count} dataset files match the release SHA-256 manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
