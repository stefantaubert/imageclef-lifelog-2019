import unittest

from src.postprocessing.OrderingTransformer import sort_dict

class UnitTests(unittest.TestCase):
    
    def test_sort_dict(self):
        subm = {
            "u1_20180505_2157_i00": 3,
            "u1_20180506_2158_i00": 2,
            "u1_20180503_0621_i01": 4,
            "u1_20180504_2154_i00": 1,
            "u1_20180507_2159_i00": 5,
        }

        res = sort_dict(subm, reverse=False)
        
        assert_res = {
            "u1_20180504_2154_i00": 1,
            "u1_20180506_2158_i00": 2,
            "u1_20180505_2157_i00": 3,
            "u1_20180503_0621_i01": 4,
            "u1_20180507_2159_i00": 5,
        }

        self.assertEqual(assert_res, res)

    def test_sort_dict_reversed(self):
        subm = {
            "u1_20180505_2157_i00": 3,
            "u1_20180506_2158_i00": 2,
            "u1_20180503_0621_i01": 4,
            "u1_20180504_2154_i00": 1,
            "u1_20180507_2159_i00": 5,
        }

        res = sort_dict(subm, reverse=True)
        
        assert_res = {
            "u1_20180507_2159_i00": 5,
            "u1_20180503_0621_i01": 4,
            "u1_20180505_2157_i00": 3,
            "u1_20180506_2158_i00": 2,
            "u1_20180504_2154_i00": 1,
        }

        self.assertEqual(assert_res, res)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
