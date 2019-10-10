import unittest
import pandas as pd
import numpy as np

from experiments.evaluation.main import evaluate
from experiments.evaluation.main import evaluate_top

class UnitTests(unittest.TestCase):
    
    def test_evaluate(self):
        gt = pd.DataFrame(
                columns=['topic_id','image_id','topic_cluster_id'], 
                data=[
                    [1,"a",1],
                    [1,"b",1],
                    [2,"d",1],
                    [2,"e",1],
                ]
            )

        subm = {
            1: {
                'a': 1.0,
            },
            2: {
                'd': 1.0,
            },
        }
        
        Xs = [0, 5]
        
        res = evaluate(subm, gt, Xs)
        print(res)
        ref_rows = [
            [1,0,1.0,1.0,1.0],
            [2,0,1.0,1.0,1.0],
            ['all',0,1.0,1.0,1.0],
            [1,5,1.0,1.0,1.0],
            [2,5,1.0,1.0,1.0],
            ['all',5,1.0,1.0,1.0],
        ]

        ref_cols = ["topic_id", "X","CR@X", "P@X", "F1@X"]

        ref = pd.DataFrame(columns=ref_cols, data=ref_rows)
        np.array_equal(ref.values,res.values)

    def test_evaluate_top(self):
        gt = pd.DataFrame(
                columns=['topic_id','image_id','topic_cluster_id'], 
                data=[
                    [1,"a",1],
                    [1,"b",1],
                    [2,"d",1],
                    [2,"e",1],
                ]
            )

        subm = {
            'a': 1.0,
        }
        
        Xs = [0, 5]
        
        res = evaluate_top(subm, 1, gt, Xs)

        ref_rows = [
            [0,1.0,1.0,1.0],
            [5,1.0,1.0,1.0],
        ]

        ref_cols = ["X","CR@X", "P@X", "F1@X"]

        ref = pd.DataFrame(columns=ref_cols, data=ref_rows)
        self.assertTrue(ref.equals(res))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
