"""Reject a false standard bound and check the all-tilt SOS certificate."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import standard_tilted_sos as sts


class TestStandardTiltedSOS(unittest.TestCase):
    def test_symbolic_family(self):
        result=sts.verify_family()
        self.assertEqual(result['status'],'PASS',result)
        self.assertTrue(result['squares_in_1_plus_AB'])
        self.assertTrue(result['attaining_strategy'])

    def test_corrupted_tilted_square_is_rejected(self):
        result=sts.verify_family(second_square_sign=-1)
        self.assertEqual(result['status'],'FAIL',result)

    def test_both_outcome_congruences(self):
        result=sts.verify_aq_block_restriction()
        self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['blocks_checked'],4)


if __name__=='__main__':unittest.main()
