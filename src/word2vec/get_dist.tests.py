import unittest

from src.word2vec.get_dist import get_dist
from src.io.reader import read_vec

class UnitTests(unittest.TestCase):
    model = read_vec(big=False)

    def test_get_dist_one_by_one(self):
        word = "test"
        other = "test"
        dist = get_dist(self.model, word, other)
        print(dist)
        self.assertLess(dist, 0)
    
    def test_get_dist_one_by_two(self):
        word = "test"
        other = "test testing"
        dist = get_dist(self.model, word, other)
        print(dist)
        self.assertLess(dist, 0.1)

    def test_get_dist_two_by_one(self):
        word = "test testing"
        other = "test"
        dist = get_dist(self.model, word, other)
        print(dist)
        self.assertLess(dist, 0.1)

    def test_get_dist_two_by_two(self):
        word = "test testing"
        other = "test testing"
        dist = get_dist(self.model, word, other)
        print(dist)
        self.assertLess(dist, 0.1)
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
