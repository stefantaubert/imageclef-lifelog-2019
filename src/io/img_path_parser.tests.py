import unittest
import pandas as pd
from src.io.reader import read_vc

from src.io.img_path_parser import get_paths
from src.io.img_path_parser import get_paths_csv

class UnitTests(unittest.TestCase):
    
    def test_get_paths_csv(self):
        root = "/unittest/"
        dummy_data = [
            ["u1_20180503_0617_i00","2018_05_03/B00001383_21I6X0_20180503_072356E.JPG", "test"],
            ["u1_20180503_0617_i01","u1_photos/2018-05-10 17.32.02.jpg", "test2"],
        ]

        dummy_vc = pd.DataFrame(columns=["image_id","image_path","attribute_top01"], data=dummy_data)
        
        paths = get_paths_csv(dummy_vc, usr=1, root=root)
        
        ref = [
            ("u1_20180503_0617_i00","/unittest/data/u1/Autographer/2018_05_03/B00001383_21I6X0_20180503_072356E.JPG"),
            ("u1_20180503_0617_i01","/unittest/data/u1/u1_photos/2018-05-10 17.32.02.jpg"),
        ]

        self.assertEqual(ref, paths)

    def test_get_paths(self):
        paths = get_paths()
        vc = read_vc()
        vc_l = len(vc.index)
        print(paths)
        self.assertEqual(vc_l, len(paths))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
