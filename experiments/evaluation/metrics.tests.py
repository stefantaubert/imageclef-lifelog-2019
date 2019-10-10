import unittest

from collections import OrderedDict

from src.evaluation.metrics import cr_x
from src.evaluation.metrics import p_x
from src.evaluation.metrics import f1_x

class UnitTests(unittest.TestCase):

    #region p
    def test_p_x_x0_empty_subm(self):
        gt = {
            1: ["d", "e"],
        }

        self.assertEqual(0, p_x(OrderedDict({ }), gt, 0))

    def test_p_x_x1_empty_subm(self):
        gt = {
            1: ["d", "e"],
        }

        self.assertEqual(0, p_x(OrderedDict({ }), gt, 1))

    def test_p_x_x0_one_miss(self):
        gt = {
            1: ["d", "e"],
        }

        subm = OrderedDict({ 
            'a': 1.0,
            'e': 1.0,
        })

        self.assertEqual(0.5, p_x(subm, gt, 0))

    def test_p_x_x0_p1(self):
        gt = {
            1: ["d", "e"],
        }

        subm = OrderedDict({ 
            'e': 1.0,
            'd': 1.0,
        })

        self.assertEqual(1, p_x(subm, gt, 0))

    def test_p_x_x_equal_clusters(self):
        gt = {
            1: ["d", "e"],
            2: ["f", "g"],
        }

        subm = OrderedDict({ 
            'd': 1.0,
            'c': 1.0,
            'f': 1.0,
            'g': 1.0,
        })

        self.assertEqual(0.5, p_x(subm, gt, 2))

    def test_p_x_x_greater_clusters(self):
        gt = {
            1: ["d", "e"],
            2: ["f", "g"],
        }

        subm = OrderedDict({ 
            'd': 1.0,
            'c': 1.0,
            'f': 1.0,
            'g': 1.0,
        })

        self.assertEqual(3.0/4.0, p_x(subm, gt, 4))

    def test_p_x_x_smaller_clusters(self):
        gt = {
            1: ["d", "e"],
            2: ["f", "g"],
        }

        subm = OrderedDict({ 
            'd': 1.0,
            'c': 1.0,
            'f': 1.0,
            'g': 1.0,
        })

        self.assertEqual(1, p_x(subm, gt, 1))

    def test_p_x_score_0(self):
        gt = {
            1: ["d", "e"],
            2: ["f", "g"],
        }

        subm = OrderedDict({ 
            'a': 1.0,
            'd': 1.0,
            'e': 1.0,
            'f': 1.0,
        })

        self.assertEqual(0, p_x(subm, gt, x=1))
    
    def test_p_x_gt_smaller_x(self):
        gt = {
            1: ["d", "e"],
            2: ["f", "g"],
        }

        subm = OrderedDict({ 
            'd': 1.0,
            'e': 1.0,
            'f': 1.0,
            'g': 1.0,
            'h': 1.0,
            'i': 1.0,
            'j': 1.0,
            'k': 1.0,
            'l': 1.0,
            'm': 1.0,
        })

        self.assertEqual(1, p_x(subm, gt, x=10))
    #endregion

    #region cr
    def test_cr_x_x_0(self):
        gt = {
            1: ["d", "e"],
        }

        subm = OrderedDict({ 
            'a': 1.0,
            'e': 1.0,
        })

        self.assertEqual(1, cr_x(subm, gt, 0))

    def test_cr_x_x_0_last_ranks(self):
        gt = {
            1: ["d", "e"],
            2: ["f", "g"],
        }

        subm = OrderedDict({ 
            'a': 1.0,
            'b': 1.0,
            'c': 1.0,
            'd': 1.0,
            'f': 1.0,
        })

        self.assertEqual(1, cr_x(subm, gt, 0))

    def test_cr_x_x0_empty_subm(self):
        gt = {
            1: ["d", "e"],
        }

        self.assertEqual(0, cr_x(OrderedDict({ }), gt, 0))

    def test_cr_x_empty_subm(self):
        gt = {
            1: ["d", "e"],
        }

        self.assertEqual(0, cr_x(OrderedDict({ }), gt, 1))

    def test_cr_x_takes_first_pred(self):
        gt = {
            1: ["d", "e"],
        }

        subm = OrderedDict({ 
            'a': 1.0,
            'e': 1.0,
        })

        self.assertEqual(0, cr_x(subm, gt, 1))

    def test_cr_x_one_cluster_missed(self):
        gt = {
            1: ["a", "b"],
            2: ["c", "d"],
        }

        subm = OrderedDict({ 
            'a': 1.0,
            'e': 1.0,
        })

        score = cr_x(subm, gt, 2)

        self.assertEqual(0.5, score)

    def test_cr_x_x_smaller_cluster_count(self):
        gt = {
            1: ["d", "e"],
            2: ["g", "h"],
        }

        subm1 = OrderedDict({ 'd': 1.0 })
        subm2 = OrderedDict({ 'h': 1.0 })

        self.assertEqual(1, cr_x(subm1, gt, 1))
        self.assertEqual(1, cr_x(subm2, gt, 1))

    def test_cr_x_x_smaller_cluster_count_two_entries(self):
        gt = {
            1: ["d", "e"],
            2: ["g", "h"],
        }

        subm = OrderedDict({ 
            'd': 1.0, 
            'f': 1.0,
        })

        self.assertEqual(1, cr_x(subm, gt, 1))

    def test_cr_x_two_equal_cluster(self):
        gt = {
            1: ["a"],
            2: ["a"],
        }

        subm = OrderedDict({ 'a': 1.0 })

        self.assertEqual(1, cr_x(subm, gt, 1))

    def test_cr_x_two_equal_cluster_x_greater_subm(self):
        gt = {
            1: ["a"],
            2: ["a"],
        }

        subm = OrderedDict({ 'a': 1.0 })

        self.assertEqual(1, cr_x(subm, gt, 2))

    def test_cr_x_x_equal_cluster_count(self):
        gt = {
            1: ["d", "e"],
        }

        subm = OrderedDict({ 'd': 1.0 })

        self.assertEqual(1, cr_x(subm, gt, 1))

    def test_cr_x_x_equal_cluster_count_missed(self):
        gt = {
            1: ["d", "e"],
        }

        subm = OrderedDict({ 'a': 1.0 })

        self.assertEqual(0, cr_x(subm, gt, 1))

    def test_cr_x_x_greater_cluster_count(self):
        gt = {
            1: ["d", "e"],
        }

        subm = OrderedDict({ 'd': 1.0 })

        self.assertEqual(1, cr_x(subm, gt, 2))
    #endregion

    #region f1
    def test_f1_x0_empty_subm(self):
        gt = {
            1: ["d", "e"],
        }

        self.assertEqual(0, f1_x(OrderedDict({ }), gt, 0))

    def test_f1_empty_subm(self):
        gt = {
            1: ["d", "e"],
        }

        self.assertEqual(0, f1_x(OrderedDict({ }), gt, 1))

    def test_f1_is_1(self):
        gt = {
            1: ["a", "b"],
            2: ["c", "d"],
        }

        subm = OrderedDict({ 
            'b': 1.0,
            'd': 1.0,
        })
        
        # p = 1, r = 1
        self.assertEqual(1, f1_x(subm, gt, 2))

    def test_f1_is_0(self):
        gt = {
            1: ["a", "b"],
            2: ["c", "d"],
        }

        subm = OrderedDict({ 
            'e': 1.0,
            'f': 1.0,
        })

        # p = 1, r = 0
        self.assertEqual(0, f1_x(subm, gt, 2))

    def test_f1_is_0_5(self):
        gt = {
            1: ["a", "b"],
            2: ["c", "d"],
        }

        subm = OrderedDict({ 
            'a': 1.0,
            'b': 1.0,
        })

        # p = 1, r = 0.5
        self.assertEqual(2*(1*0.5)/(1+0.5), f1_x(subm, gt, 2))
    #endregion

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
