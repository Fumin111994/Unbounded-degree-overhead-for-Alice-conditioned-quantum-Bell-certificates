"""M2.6 tests: exact level-one family results for tilted CHSH."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FAMILY_DIR = ROOT / "artifacts" / "m26" / "family"
ANCHORS_DIR = ROOT / "artifacts" / "m26" / "anchors"


class StandardFamilyTests(unittest.TestCase):
    """omega_std^1(alpha) = 2 sqrt(2) + alpha for all 0 <= alpha < 2."""

    def test_family_script_verifies(self) -> None:
        # The script is solver-free: it verifies the parametric SOS identity
        # and the exact witness in symbolic alpha.
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_level1_family.py")],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((FAMILY_DIR / "standard_family.json").read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["witness"]["psd_exact"])


class OneSidedAnchorTests(unittest.TestCase):
    """Anchor certificates and the exact covering of (0, 2)."""

    def test_coverage_report(self) -> None:
        path = ANCHORS_DIR / "coverage.json"
        if not path.exists():
            self.skipTest("run scripts/run_level1_anchors.py first")
        coverage = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(coverage["status"], "PASS", coverage["failures"])
        self.assertEqual(coverage["covered_interval"], "(0, 2)")
        endpoint = coverage["endpoint_alpha0"]["certificate"]
        self.assertEqual(endpoint["status"], "PASS")
        self.assertTrue(endpoint["nice"])
        for anchor in coverage["anchors"]:
            self.assertEqual(anchor["verification"], "PASS")
            self.assertTrue(anchor["strictly_below_standard"])

    def test_alpha_one_half_anchor_reverification(self) -> None:
        from run_exact_level1 import verify_onesided_artifact

        path = ANCHORS_DIR / "anchor_1_2.json"
        if not path.exists():
            self.skipTest("run scripts/run_level1_anchors.py first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = verify_onesided_artifact(payload)
        self.assertEqual(report["status"], "PASS", report["failures"])


if __name__ == "__main__":
    unittest.main()
