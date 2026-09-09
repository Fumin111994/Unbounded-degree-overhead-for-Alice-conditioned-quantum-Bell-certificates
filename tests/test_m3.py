"""M3 tests: benchmark search artifacts and the exact level-2 separation."""

import json
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SEARCH_RESULTS = [
    ROOT / "artifacts" / "m3" / "search" / "results.json",
    ROOT / "artifacts" / "m3" / "search_s777" / "results.json",
    ROOT / "artifacts" / "m3" / "search_s31337" / "results.json",
]
EXACT_ARTIFACT = ROOT / "artifacts" / "m3" / "exact" / "search_s31337_308.json"


class M3SearchTests(unittest.TestCase):
    def test_search_reports(self) -> None:
        for path in SEARCH_RESULTS:
            if not path.exists():
                self.skipTest(f"{path} missing; run scripts/run_m3_search.py")
        for path in SEARCH_RESULTS:
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["flag_counts"].get("solver_failure", 0), 0)
            for record in report["records"]:
                if record["flag"] in ("SEPARATION_CANDIDATE", "TRANSPORT_EXAMPLE"):
                    self.assertTrue(record["novel"])
                    self.assertTrue(record["nonlocal"])
                    self.assertLess(abs(record["std2"] - record["os3"]), 1e-5)

    def test_exact_level_two_separation_artifact(self) -> None:
        if not EXACT_ARTIFACT.exists():
            self.skipTest("run scripts/run_m3_exactify.py first")
        from run_m3_exactify import verify_artifact

        payload = json.loads(EXACT_ARTIFACT.read_text(encoding="utf-8"))
        report = verify_artifact(payload)
        self.assertEqual(report["status"], "PASS", report["failures"])

    def test_o3_membership_artifact(self) -> None:
        path = ROOT / "artifacts" / "m3" / "exact" / "o3_308.json"
        if not (path.exists() and EXACT_ARTIFACT.exists()):
            self.skipTest("run the O_3 certificate computation first")
        from fractions import Fraction

        from npa2 import exact
        from run_m3_exactify import rebuild_functional

        main_payload = json.loads(EXACT_ARTIFACT.read_text(encoding="utf-8"))
        functional = rebuild_functional(main_payload)
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = list(payload["certificate"].keys())
        values = []
        for name in names:
            num, _, den = payload["certificate"][name].partition("/")
            values.append(Fraction(int(num), int(den or "1")))
        s_blocks, lam, nu, mu = exact.unpack_onesided_unknowns(names, values)
        report = exact.verify_onesided_dual(
            functional, 3, Fraction(501, 200), s_blocks, lam, nu, mu,
            enforce_probability_positivity=False,
        )
        self.assertEqual(report["status"], "PASS", report["failures"])


if __name__ == "__main__":
    unittest.main()
