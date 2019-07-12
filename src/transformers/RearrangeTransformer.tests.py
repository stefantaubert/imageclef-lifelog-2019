import unittest

from collections import OrderedDict

from src.postprocessing.RearrangeTransformer import get_day_from_img_id
from src.postprocessing.RearrangeTransformer import rearrange

class UnitTests(unittest.TestCase):
    
    def test_get_day_from_img_id(self):
        img_id = "u1_20180503_0619_i08"
        day = get_day_from_img_id(img_id)
        self.assertEqual("20180503", day)

    def test_rearrange_equal(self):
        subm = OrderedDict({
            "u1_20180504_2154_i00": 1,
            "u1_20180506_2158_i00": 2,
            "u1_20180505_2157_i00": 3,
            "u1_20180503_0621_i01": 4,
            "u1_20180507_2159_i00": 5,
        })

        res = rearrange(subm, per_day=1)

        self.assertEqual(subm, res)

    def test_rearrange_simple(self):
        subm = OrderedDict({
            "___20180504________1": 1,
            "___20180504________2": 2,
            "___20180506________3": 3,
            "___20180507________4": 4,
            "___20180507________5": 5,
        })

        res = rearrange(subm, per_day=1)

        assert_res = OrderedDict({  
            "___20180504________1": 1,
            "___20180506________3": 3,
            "___20180507________4": 4,
            "___20180504________2": 2,
            "___20180507________5": 5,
        })

        self.assertEqual(assert_res, res)

    def test_rearrange_extended(self):
        subm = OrderedDict({
            "___20180504________1": 1,
            "___20180504________2": 2,
            "___20180506________3": 3,
            "___20180506________4": 3,
            "___20180507________5": 4,
            "___20180507________6": 5,
            "___20180504________7": 2,
            "___20180504________8": 2,
            "___20180504________9": 2,
        })

        res = rearrange(subm, per_day=1)

        assert_res = OrderedDict({  
            "___20180504________1": 1,
            "___20180506________3": 3,
            "___20180507________5": 4,
            "___20180504________2": 2,
            "___20180506________4": 3,
            "___20180507________6": 5,
            "___20180504________7": 2,
            "___20180504________8": 2,
            "___20180504________9": 2,
        })

        self.assertEqual(assert_res, res)

    def test_rearrange_two_per_day(self):
        subm = OrderedDict({
            "___20180504________1": 1,
            "___20180504________2": 2,
            "___20180506________3": 3,
            "___20180506________4": 3,
            "___20180506________5": 3,
            "___20180507________6": 4,
            "___20180507________7": 5,
            "___20180507________8": 5,
            "___20180507________9": 5,
            "___20180504________10": 2,
            "___20180504________11": 2,
            "___20180504________12": 2,
        })

        res = rearrange(subm, per_day=2)

        assert_res = OrderedDict({  
            "___20180504________1": 1,
            "___20180504________2": 2,
            "___20180506________3": 3,
            "___20180506________4": 3,
            "___20180507________6": 4,
            "___20180507________7": 5,
            "___20180504________10": 2,
            "___20180504________11": 2,
            "___20180506________5": 3,
            "___20180507________8": 5,
            "___20180507________9": 5,
            "___20180504________12": 2,
        })

        self.assertEqual(assert_res, res)

    def test_rearrange_zero(self):
        subm = OrderedDict({
            "___20180504________1": 1,
            "___20180504________2": 2,
            "___20180506________3": 3,
            "___20180506________4": 3,
            "___20180506________5": 3,
            "___20180507________6": 4,
            "___20180507________7": 5,
            "___20180507________8": 5,
            "___20180507________9": 5,
            "___20180504________10": 2,
            "___20180504________11": 2,
            "___20180504________12": 2,
        })

        res = rearrange(subm, per_day=0)
     
        self.assertEqual(subm, res)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
