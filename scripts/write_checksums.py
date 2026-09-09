#!/usr/bin/env python3
"""M7: write SHA-256 checksum manifests for the M2.5+ artifact trees."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TREES = [
    "artifacts/prl/interval",
    "artifacts/prl/standard_tilted",
    "artifacts/prl/quantum_face",
    "artifacts/m25/exact",
    "artifacts/m25/fulloutcome",
    "artifacts/m26/grid",
    "artifacts/m26/family",
    "artifacts/m26/anchors",
    "artifacts/m3/exact",
    "artifacts/m3/candidates",
    "artifacts/m65",
    "artifacts/m6",
    "artifacts/figures",
    "figures",
]


def main() -> int:
    for tree in TREES:
        directory = ROOT / tree
        if not directory.exists():
            print(f"skip missing {tree}")
            continue
        lines = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines.append(f"{digest}  {path.relative_to(directory)}")
        (directory / "checksums.sha256").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
        print(f"{tree}: {len(lines)} files checksummed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
