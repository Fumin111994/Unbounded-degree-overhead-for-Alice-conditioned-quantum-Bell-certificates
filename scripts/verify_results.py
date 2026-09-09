#!/usr/bin/env python3
"""Recheck exported M2 artifacts without running an SDP solver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.verify import verify_artifacts  # noqa: E402


def main() -> int:
    output = ROOT / "artifacts" / "m2" / "canonical"
    report = verify_artifacts(output)
    (output / "verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
