"""Solver-free certificate re-verification as a unit test.

Runs scripts/verify_certificates.py's main() in-process; all SDP calls are
patched to raise inside the verifier itself, so this test never touches a
solver.  Requires the artifact tree (tests skip if artifacts are absent).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class CertificateVerificationTests(unittest.TestCase):
    def test_verify_certificates_all_pass(self) -> None:
        if not (ROOT / "artifacts" / "m3" / "exact" / "search_s31337_308.json").exists():
            self.skipTest("artifact tree not present")
        import verify_certificates

        rc = verify_certificates.main([])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
