import unittest
import pandas as pd

from src.common.gt_converter import gt_to_dict

class UnitTests(unittest.TestCase):
    
    def test_extract_date(self):
        data = [
            [1,'u1_20180528_1816_i00',1],
            [1,'u1_20180528_1816_i02',1],
            [2,'u1_20180508_1106_i01',1],
            [2,'u1_20180508_1106_i00',1],
            [2,'u1_20180523_0341_i04',2],
        ]

        gt = pd.DataFrame(columns=["topic_id", "image_id", "topic_cluster_id"], data=data)

        gt_dict = gt_to_dict(gt)

        assert_res = {
            1: {
                1: ['u1_20180528_1816_i00', 'u1_20180528_1816_i02'],
            },
            2: {
                1: ['u1_20180508_1106_i01', 'u1_20180508_1106_i00'],
                2: ['u1_20180523_0341_i04'],
            }
        }

        self.assertEqual(assert_res, gt_dict)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
