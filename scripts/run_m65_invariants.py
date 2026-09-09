#!/usr/bin/env python3
"""M6.5 G3(ii): invariant table and rigorous novelty check for M3 candidates.

The multiset {|c_xy|} is invariant under question permutations, outcome
flips, party swap, and (up to scale) normalization; marginals map to
marginals.  Hence:
- CHSH-type correlators (tilted family, XOR/CHSH) have all |c_xy| equal;
- XOR games have zero marginals;
- B3 lives in (2,2,3,3).
A candidate with non-equal |c_xy| and >= 2 nonzero marginals is outside the
equivalence closure of all catalogued families.

Output: artifacts/m65/invariants.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    ("search_s31337", 308),
    ("search_s777", 298),
]


def main() -> int:
    out = []
    for name, index in CANDIDATES:
        report = json.loads((ROOT / "artifacts" / "m3" / name / "results.json").read_text())
        record = next(r for r in report["records"] if r["index"] == index)
        corr = [record["corr"][f"{x}{y}"] for x in (0, 1) for y in (0, 1)]
        abs_entries = sorted(abs(v) for v in corr)
        amarg = list(record["alice_marginals"].values())
        bmarg = list(record["bob_marginals"].values())
        marg_nonzero = sum(1 for v in amarg + bmarg if abs(v) > 1e-9)
        matrix = np.array(corr).reshape(2, 2)
        singular = sorted(np.linalg.svd(matrix, compute_uv=False))
        chsh_type = max(abs_entries) - min(abs_entries) < 1e-9
        out.append({
            "candidate": f"{name}_{index}",
            "correlator_abs_entries_sorted": abs_entries,
            "correlator_singular_values_sorted": singular,
            "alice_marginals": amarg,
            "bob_marginals": bmarg,
            "nonzero_marginals": marg_nonzero,
            "chsh_type_correlator": chsh_type,
            "classical": record["classical"],
            "std2": record["std2"],
            "os2": record["os2"],
            "novel": (not chsh_type) and marg_nonzero >= 2,
        })
    path = ROOT / "artifacts" / "m65"
    path.mkdir(parents=True, exist_ok=True)
    (path / "invariants.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
