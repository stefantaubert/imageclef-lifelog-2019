import random
import unittest

from src.common.train_test_split import train_test_split_list
from src.common.train_test_split import train_test_split_gt

def dummy_sample(list, n):
    return list[:n]

class UnitTests(unittest.TestCase):
    
    def test_train_test_split_list_0_2(self):
        lst = [1, 2, 3]
        
        train, test = train_test_split_list(lst, 0.2, False, 0)
       
        assert_train = [1, 2]
        assert_test = [3]

        self.assertEqual(assert_train, train)
        self.assertEqual(assert_test, test)
    
    def test_train_test_split_list(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        train, test = train_test_split_list(lst, 0.5, False, 0)
       
        assert_train = [1, 2, 3, 4, 5]
        assert_test = [6, 7, 8, 9, 10]

        self.assertEqual(assert_train, train)
        self.assertEqual(assert_test, test)
    
    def test_train_test_split_list_0(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        train, test = train_test_split_list(lst, 0.0, False, 0)
       
        assert_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert_test = []

        self.assertEqual(assert_train, train)
        self.assertEqual(assert_test, test)
    
    def test_train_test_split_list_1(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        train, test = train_test_split_list(lst, 1.0, False, 0)
       
        assert_train = []
        assert_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.assertEqual(assert_train, train)
        self.assertEqual(assert_test, test)
    
    def test_train_test_split_gt(self):
        gt = {
            1: {   
                1: ["d", "e"],
                2: ["f", "g"],
            },
            2: {
                1: ["a", "d"],
                2: ["b", "c"],
            }
        }

        train, test = train_test_split_gt(gt, 0.5, False, 0)

        assert_train = {
            1: ["d", "f"],
            2: ["a", "b"],
        }

        assert_test = {
            1: ["e", "g"],
            2: ["d", "c"],
        }

        self.assertEqual(assert_train, train)
        self.assertEqual(assert_test, test)

    def test_train_test_split_gt_0(self):
        gt = {
            1: {   
                1: ["d", "e"],
                2: ["f", "g"],
            },
            2: {
                1: ["a", "d"],
                2: ["b", "c"],
            }
        }

        train, test = train_test_split_gt(gt, 0.0, False, 0)

        assert_train = {
            1: ["d", "e", "f", "g"],
            2: ["a", "d", "b", "c"],
        }

        assert_test = { 1: [], 2: [] }

        self.assertEqual(assert_train, train)
        self.assertEqual(assert_test, test)

    def test_train_test_split_gt_1(self):
        gt = {
            1: {   
                1: ["d", "e"],
                2: ["f", "g"],
            },
            2: {
                1: ["a", "d"],
                2: ["b", "c"],
            }
        }

        train, test = train_test_split_gt(gt, 1.0, False, 0)

        assert_train = { 1: [], 2: [] }

        assert_test = {
            1: ["d", "e", "f", "g"],
            2: ["a", "d", "b", "c"],
        }

        self.assertEqual(assert_train, train)
        self.assertEqual(assert_test, test)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
