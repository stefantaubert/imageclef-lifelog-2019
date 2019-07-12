import unittest

from datetime import date
from src.common.helper import extract_date
from src.common.helper import extract_time
from src.common.helper import extract_datetime
from src.common.helper import extract_autographer_datetime
from src.common.helper import extract_personal_datetime
from src.common.helper import time_to_datetime
from src.common.helper import round_to_full_minute
from src.common.helper import delete_minute_in_time

class UnitTests(unittest.TestCase):
    
    def test_extract_date(self):
        extracted = extract_date("20180503_0000_UTC")
        self.assertEqual(3, extracted.day)
        self.assertEqual(5, extracted.month)
        self.assertEqual(2018, extracted.year)

    def test_extract_time(self):
        extracted = extract_time("20180503_2359_UTC")
        self.assertEqual(59, extracted.minute)
        self.assertEqual(23, extracted.hour)

    def test_extract_datetime(self):
        extracted = extract_datetime("20180503_2359_UTC")
        self.assertEqual(3, extracted.day)
        self.assertEqual(5, extracted.month)
        self.assertEqual(2018, extracted.year)
        self.assertEqual(59, extracted.minute)
        self.assertEqual(23, extracted.hour)

    def test_extract_autographer_datetime(self):
        extracted = extract_autographer_datetime("B00008061_21I6X0_20180506_105818E.JPG")
        self.assertEqual(6, extracted.day)
        self.assertEqual(5, extracted.month)
        self.assertEqual(2018, extracted.year)
        self.assertEqual(10, extracted.hour)
        self.assertEqual(58, extracted.minute)
        self.assertEqual(18, extracted.second)

    def test_extract_personal_datetime(self):
        extracted = extract_personal_datetime("2018-05-10 08.06.48.jpg")
        self.assertEqual(10, extracted.day)
        self.assertEqual(5, extracted.month)
        self.assertEqual(2018, extracted.year)
        self.assertEqual(8, extracted.hour)
        self.assertEqual(6, extracted.minute)
        self.assertEqual(48, extracted.second)

    def test_time_to_datetime(self):
        extracted = extract_time("20180503_2359_UTC")
        dt = time_to_datetime(extracted)
        self.assertEqual(1, dt.day)
        self.assertEqual(1, dt.month)
        self.assertEqual(1970, dt.year)
        self.assertEqual(59, dt.minute)
        self.assertEqual(23, dt.hour)
        
    def test_delete_minute_in_time(self):
        extracted = extract_time("20180503_2359_UTC")
        dt = delete_minute_in_time(extracted)
        self.assertEqual(0, dt.minute)
        self.assertEqual(23, dt.hour)

    def test_round_to_full_minute(self):
        extracted = extract_time("20180503_2359_UTC")
        dt = round_to_full_minute(extracted)
        self.assertEqual(50, dt.minute)
        self.assertEqual(23, dt.hour)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)