import unittest

from experiments.helper import flatten

class UnitTests(unittest.TestCase):
    
    def test_sort_dict(self):
        opts = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        d = flatten(opts)

        assert_res = {
            "a": 1,
            "b_c": 2,
            "b_d_e": 3,
        }

        self.assertEqual(assert_res, d)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
