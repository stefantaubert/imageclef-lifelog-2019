import unittest
from src.data.DaytimeData import __get_day_time__

class UnitTests(unittest.TestCase):
    
    def test_get_day_time_0819(self):
        self.assertEqual("morning", __get_day_time__("20180503_0819"))

    def test_get_day_time_0000(self):
        self.assertEqual("night", __get_day_time__("20180503_0000"))

    def test_get_day_time_0300(self):
        self.assertEqual("night", __get_day_time__("20180503_0300"))

    def test_get_day_time_2200(self):
        self.assertEqual("night", __get_day_time__("20180503_2200"))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
