#!/usr/bin/env python3
"""M7: write SHA-256 checksum manifests for the M2.5+ artifact trees."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TREES = [
    "data/analytic/interval",
    "data/analytic/standard_tilted",
    "data/analytic/quantum_face",
    "data/level_one_exact/exact",
    "data/level_one_exact/fulloutcome",
    "data/level_one_family/grid",
    "data/level_one_family/family",
    "data/level_one_family/anchors",
    "data/finite_separations/exact",
    "data/finite_separations/candidates",
    "data/contact_geometry",
    "data/conditioned_closure",
    "data/figures",
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
