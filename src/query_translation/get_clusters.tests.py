import unittest

from src.query_translation.get_clusters import fclusters_to_indicies

class UnitTests(unittest.TestCase):

    def test_fclusters_to_indicies(self):
        fclusters = [1,1,2,2,3,4,1,2]

        ind = fclusters_to_indicies(fclusters)

        assert_res = [
            [0,1,6],
            [2,3,7],
            [4],
            [5],
        ]

        self.assertEqual(assert_res, ind)

    def test_fclusters_to_indicies_one_entry(self):
        fclusters = [1]

        ind = fclusters_to_indicies(fclusters)

        assert_res = [
            [0],
        ]

        self.assertEqual(assert_res, ind)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
