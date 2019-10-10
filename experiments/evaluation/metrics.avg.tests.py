import unittest

from src.evaluation.metrics import avg_cr_x
from src.evaluation.metrics import avg_p_x
from src.evaluation.metrics import avg_f1_x

class UnitTests(unittest.TestCase):

    def test_avg_f1_is_0_5(self):
        gt = {
            1: {   
                1: ["d", "e"],
                2: ["f", "g"],
            },
            2: {
                1: ["d", "e"],
                2: ["f", "g"],
            }
        }

        subm = {
            1: {
                'd': 1.0,
                'f': 1.0,
            },
            2: {
                'a': 1.0,
                'b': 1.0,
            },
        }

        # 1: 1, 2: 0
        self.assertEqual(0.5, avg_f1_x(subm, gt, 2))

    def test_avg_p_x_score_0(self):
        gt = {
            1: {   
                1: ["d", "e"],
                2: ["f", "g"],
            },
            2: {
                1: ["d", "e"],
                2: ["f", "g"],
            }
        }

        subm = {
            1: {
                'a': 1.0,
                'd': 1.0,
            },
            2: {
                'a': 1.0,
                'e': 1.0,
            },
        }

        self.assertEqual(0, avg_p_x(subm, gt, 1))

    def test_avg_p_x_score_0_5(self):
        gt = {
            1: {   
                1: ["d", "e"],
                2: ["f", "g"],
            },
            2: {
                1: ["d", "e"],
                2: ["f", "g"],
            }
        }

        subm = {
            1: {
                'a': 1.0,
                'd': 1.0,
            },
            2: {
                'a': 1.0,
                'e': 1.0,
            },
        }

        self.assertEqual(0.5, avg_p_x(subm, gt, 2))

    def test_avg_cr_x_perfect_score(self):
        gt = {
            1: {
                1: ["a", "b"],
            },
            2: {
                1: ["d", "e"],
            },
        }

        subm = {
            1: {
                'a': 1.0,
            },
            2: {
                'd': 1.0,
            },
        }

        score = avg_cr_x(subm, gt, 1)

        self.assertEqual(1.0, score)

    def test_avg_cr_x_perfect_one_wrong(self):
        gt = {
            1: {
                1: ["a", "b"],
            },
            2: {
                1: ["d", "e"],
            },
        }

        subm = {
            1: {
                'a': 1.0,
            },
            2: {
                'f': 1.0,
            },
        }

        score = avg_cr_x(subm, gt, 1)

        self.assertEqual(0.5, score)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
