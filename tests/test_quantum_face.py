"""Independent acceptance and rejection tests for optimal-bound certificates."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class QuantumFaceTests(unittest.TestCase):
    def verifier(self):
        self.assertIsNotNone(importlib.util.find_spec('quantum_face'),
                             'An independent exact verifier is required')
        from quantum_face import verify_certificate
        return verify_certificate

    def payload(self):
        return json.loads((ROOT / 'artifacts/prl/quantum_face/generic_face_1_L2.json').read_text())

    def test_five_certificates_without_solver(self):
        verify = self.verifier()
        import cvxpy as cp
        paths = list((ROOT / 'artifacts/prl/quantum_face').glob('*.json'))
        self.assertEqual(len(paths), 5)
        with patch.object(cp.Problem, 'solve', side_effect=AssertionError('solver forbidden')):
            for path in paths:
                report = verify(json.loads(path.read_text()))
                self.assertEqual(report['status'], 'PASS', (path.name, report))
                self.assertTrue(report['reduced_grams_positive_definite'])

    def test_bound_below_quantum_is_rejected(self):
        verify = self.verifier()
        data = self.payload()
        data['bound'] = '3'
        self.assertEqual(verify(data)['status'], 'FAIL')

    def test_corrupted_identity_is_rejected(self):
        verify = self.verifier()
        data = self.payload()
        index = data['names'].index('nu')
        data['u'][index] = '(' + data['u'][index] + ')+1'
        self.assertEqual(verify(data)['status'], 'FAIL')

    def test_indefinite_reduced_gram_is_rejected(self):
        verify = self.verifier()
        data = self.payload()
        data['H'][0][0][0] = '-1'
        self.assertEqual(verify(data)['status'], 'FAIL')

    def test_missing_block_is_rejected(self):
        verify = self.verifier()
        data = self.payload()
        data['K'].pop()
        self.assertEqual(verify(data)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
