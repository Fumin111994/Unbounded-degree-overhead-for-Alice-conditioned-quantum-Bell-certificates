"""Interval verification must reject failures between the endpoints."""
import copy
import sys
import unittest
from pathlib import Path
import sympy as sp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import quantum_interval as qi


class TestQuantumInterval(unittest.TestCase):
    def test_interval_certificate(self):
        result=qi.verify_all()
        self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['closure_interval_contains'],['13/10','3/2'])
        self.assertEqual(result['bernstein_matrices_checked'],84)
        self.assertEqual(result['bernstein_positive_definite_matrices'],84)
        self.assertEqual(result['quantum_word_maps_checked'],4)

    def test_full_rank_is_not_enough_for_a_quantum_kernel(self):
        payload=qi.load_certificate()
        kernels=[qi._matrix(m,(7,5)) for m in payload['K']]
        kernels[0][0,0]+=1
        lo,hi=map(sp.Rational,payload['parameter_interval'])
        with self.assertRaisesRegex(ValueError,'quantum kernel'):
            qi.verify_quantum_kernels(kernels,lo,hi)

    def test_identity_error_vanishing_at_every_search_node(self):
        payload=copy.deepcopy(qi.load_certificate())
        lo,hi=map(sp.Rational,payload['parameter_interval'])
        nodes=[lo+(hi-lo)*sp.Rational(j,8) for j in range(9)]
        error=sp.prod(qi.U-node for node in nodes)
        self.assertTrue(all(error.subs(qi.U,node)==0 for node in nodes))
        i=payload['names'].index('nu')
        payload['u'][i]='('+payload['u'][i]+')+('+str(error)+')'
        result=qi.verify_certificate(payload)
        self.assertEqual(result['status'],'FAIL',result)
        self.assertTrue(any('identity' in f for f in result['failures']))

    def test_interior_negative_polynomial(self):
        u=qi.U
        lo,hi=sp.Rational(43,10),sp.Rational(447,100)
        mid=(lo+hi)/2
        poly=(u-mid)**2-((hi-lo)/4)**2
        self.assertGreater(poly.subs(u,lo),0)
        self.assertGreater(poly.subs(u,hi),0)
        result=qi.check_bernstein_psd([sp.eye(5)*poly],lo,hi)
        self.assertEqual(result['status'],'FAIL',result)

    def test_changed_quantum_bound(self):
        payload=copy.deepcopy(qi.load_certificate())
        payload['q']='('+payload['q']+')+1/1000'
        self.assertEqual(qi.verify_certificate(payload)['status'],'FAIL')

    def test_internal_pole_is_rejected(self):
        u=qi.U
        lo,hi=sp.Rational(43,10),sp.Rational(447,100)
        with self.assertRaises(ValueError):
            qi.check_no_poles([1/(u-(lo+hi)/2)],lo,hi)

    def test_changed_identity(self):
        payload=copy.deepcopy(qi.load_certificate())
        i=payload['names'].index('nu')
        payload['u'][i]='('+payload['u'][i]+')+1'
        result=qi.verify_certificate(payload)
        self.assertEqual(result['status'],'FAIL',result)
        self.assertTrue(any('identity' in f for f in result['failures']))


if __name__=='__main__':unittest.main()
