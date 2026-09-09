"""Independent checks of the strengthened all-level construction."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import cvxpy as cp
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import higher_level_witness as hw


class FejerWitnessTests(unittest.TestCase):
    def test_original_projector_constraints_and_separation(self):
        with patch.object(cp.Problem,'solve',side_effect=RuntimeError('disabled')):
            for k in [1,2,3,5,8]:
                p=hw.fejer_certificate(k)
                result=hw.verify_certificate(p)
                self.assertEqual(result['status'],'PASS',result)
                self.assertEqual(sp.Rational(p['r']),sp.Rational(1,8*k*k+2))
                self.assertEqual(sp.Rational(result['h']),1-2*sp.Rational(p['r']))
                self.assertEqual(sp.Rational(result['g']),2+4*sp.Rational(p['r']))

    def test_functional_gram_is_moving_average_gram(self):
        for k in [1,2,3,6]:
            w=hw.words(k)
            path=[tuple((a+j)%2 for j in range(t))
                  for a,ts in [(0,range(k,0,-1)),(1,range(1,k+1))] for t in ts]
            path.insert(k,())
            order=[w.index(t) for t in path]
            blocks=hw.fejer_blocks(k)
            M=(blocks[0]+blocks[1]).extract(order,order)
            N=2*k
            R=sp.Matrix(N+1,2*N,lambda i,j:int(bool(i<=j<i+N)))
            self.assertEqual(M,R*R.T/N)
            self.assertEqual(R.rank(),N+1)

    def test_rank_one_threshold_is_saturated(self):
        for k in [1,2,4]:
            blocks=hw.fejer_blocks(k)
            v=sp.Matrix([(-1)**len(w) for w in hw.words(k)])
            M=blocks[0]+blocks[1]
            z=M.inv()*v
            self.assertEqual(v.dot(z),8*k*k+2)
            self.assertEqual(blocks[0]*z,sp.zeros(2*k+1,1))
            blocks[0]-=blocks[1]/100
            self.assertLess((z.T*blocks[0]*z)[0],0)
            p=hw.payload_from_blocks(2-sp.Rational(8,8*k*k+2),k,blocks)
            self.assertEqual(hw.verify_certificate(p)['status'],'FAIL')

    def test_endpoint_extension_is_exact_and_overlaps_fan(self):
        r=sp.Rational(1,34)
        threshold=16*r/(1+4*r-4*r*r)
        self.assertEqual(threshold,sp.Rational(68,161))
        self.assertEqual(2-threshold,sp.Rational(254,161))
        # Existing fan contains alpha=8/5; the analytic interval does too.
        self.assertLess(2-threshold,sp.Rational(8,5))
        p=hw.fejer_certificate(2)
        p['alpha']='8/5'
        p['gap_lower_bound']='1/10000'
        self.assertEqual(hw.verify_certificate(p)['status'],'PASS')

    def test_every_symbolic_identity_is_checked_without_solvers(self):
        with patch.object(cp.Problem,'solve',side_effect=RuntimeError('disabled')):
            result=hw.verify_fejer_family()
        self.assertEqual(result['status'],'PASS',result)
        self.assertGreaterEqual(result['symbolic_identities_checked'],6)

    def test_positive_tolerance_obstruction(self):
        p=hw.fejer_certificate(3)
        result=hw.verify_certificate(p)
        self.assertEqual(result['alpha'],'70/37')
        self.assertEqual(result['objective_exact'],'5332/1369')
        delta=sp.Rational(p['gap_lower_bound'])
        w=sp.Rational(result['objective_exact'])
        self.assertGreater((w-delta)**2-sp.Rational(result['quantum_bound_squared']),0)


if __name__=='__main__':unittest.main()
