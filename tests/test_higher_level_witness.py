"""Soundness checks for high-level counterexamples to degree conversion."""
import copy
import importlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import cvxpy as cp
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]


class HigherLevelWitnessTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('higher_level_witness'),
                             'Higher-level exact verifier is missing')
        return importlib.import_module('higher_level_witness')

    def test_trace_reference_is_strictly_feasible_in_original_basis(self):
        m=self.module()
        for level in [3,4]:
            T=m.projector_transform(level)
            n=2*level+1
            self.assertNotEqual(T.det(),0)
            blocks=[sp.eye(n)/2 for _ in range(4)]
            p=m.payload_from_blocks(sp.Rational(9,5),level,blocks)
            self.assertEqual(m.verify_certificate(p,require_separation=False)['status'],'PASS')

    def test_missing_entries_are_not_silently_zero_filled(self):
        m=self.module()
        p=m.payload_from_blocks(sp.Rational(9,5),3,[sp.eye(7)/2 for _ in range(4)])
        del p['block_entries']['0:0:0:1']
        self.assertEqual(m.verify_certificate(p,require_separation=False)['status'],'FAIL')

    def test_false_separation_is_rejected(self):
        m=self.module()
        p=m.payload_from_blocks(sp.Rational(9,5),3,[sp.eye(7)/2 for _ in range(4)])
        self.assertEqual(m.verify_certificate(p)['status'],'FAIL')

    def test_stored_counterexamples_are_verified_without_solvers(self):
        m=self.module()
        with patch.object(cp.Problem,'solve',side_effect=RuntimeError('disabled')):
            result=m.verify_all()
        self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['levels_checked'],[3,4])

    def test_damaged_involution_gram_is_rejected(self):
        m=self.module()
        p=m.payload_from_blocks(sp.Rational(9,5),3,[sp.eye(7)/2 for _ in range(4)])
        p['involution_grams'][0][0][0]='-1'
        self.assertEqual(m.verify_certificate(p,require_separation=False)['status'],'FAIL')

    def test_analytic_counterexample_has_simple_exact_objective(self):
        m=self.module()
        self.assertTrue(hasattr(m,'analytic_certificate'),'Analytic family is missing')
        p=m.analytic_certificate(3)
        r=m.verify_certificate(p)
        self.assertEqual(r['status'],'PASS',r)
        self.assertEqual(r['objective_exact'],'1723/441')
        self.assertEqual(r['alpha'],'40/21')

    def test_excess_rank_one_subtraction_is_rejected(self):
        m=self.module()
        self.assertTrue(hasattr(m,'analytic_blocks'),'Analytic matrix construction is missing')
        blocks=m.analytic_blocks(3)
        blocks[0]-=blocks[1]
        blocks[1]*=2
        p=m.payload_from_blocks(sp.Rational(40,21),3,blocks)
        self.assertEqual(m.verify_certificate(p)['status'],'FAIL')

    def test_unbounded_family_symbolic_identities(self):
        m=self.module()
        self.assertTrue(hasattr(m,'verify_analytic_family'),'Symbolic all-level check is missing')
        with patch.object(cp.Problem,'solve',side_effect=RuntimeError('disabled')):
            r=m.verify_analytic_family()
        self.assertEqual(r['status'],'PASS',r)
        self.assertEqual(r['symbolic_identities_checked'],4)


if __name__=='__main__':unittest.main()
