import numpy as np
import pandas as pd
import unittest
import math

from src.detecting.DetectorBase import DetectorBase
from src.io.reader import read_vc

class TestDetector(DetectorBase):
    def detect_image(self, file):
        return [(file, 0.9, (1, 2, 3, 4)), ("test", 0.8, (4, 3, 2, 1))]

class TestDetectorBytes(DetectorBase):
    def detect_image(self, file):
        return [("test".encode('utf-8'), 0.9, (1, 2, 3, 4))]

class TestDetectorDiffResLen(DetectorBase):
    def detect_image(self, file):
        if "1" in file:
            return [(file, 0.9, (1, 2, 3, 4)), ("test", 0.8, (4, 3, 2, 1))]
        else:
            return [(file, 0.9, (1, 2, 3, 4))]

class UnitTests(unittest.TestCase):
    
    def test_detect_images_empty(self):
        d = TestDetector()

        res = d.detect_images([])

        self.assertEqual([], res)

    def test_detect_images_two_entries(self):
        d = TestDetector()
        img_paths = [
            ("id1", "testpath1.jpg"),
            ("id2", "testpath2.jpg"),
        ]

        res = d.detect_images(img_paths)

        ref_rows = [
            ["id1", "testpath1.jpg", 0.9, "1 2 3 4", "test", 0.8, "4 3 2 1"],
            ["id2", "testpath2.jpg", 0.9, "1 2 3 4", "test", 0.8, "4 3 2 1"],
        ]

        self.assertEqual(ref_rows, res)

    def test_detect_images_bytes(self):
        d = TestDetectorBytes()
        img_paths = [
            ("id1", "testpath1.jpg"),
        ]

        res = d.detect_images(img_paths)

        ref_rows = [
            ["id1", "test", 0.9, "1 2 3 4"],
        ]

        self.assertEqual(ref_rows, res)

    def test_detect_images_diff_len(self):
        d = TestDetectorDiffResLen()
        img_paths = [
            ("id1", "testpath1.jpg"),
            ("id2", "testpath2.jpg"),
        ]

        res = d.detect_images(img_paths)

        ref_rows = [
            ["id1", "testpath1.jpg", 0.9, "1 2 3 4", "test", 0.8, "4 3 2 1"],
            ["id2", "testpath2.jpg", 0.9, "1 2 3 4"],
        ]

        self.assertEqual(ref_rows, res)

    def test_detect_images_df(self):
        d = TestDetector()
        img_paths = [
            ("id1", "testpath1.jpg"),
            ("id2", "testpath2.jpg"),
        ]

        res = d.detect_images_df(img_paths)

        ref_rows = [
            ["id1", "testpath1.jpg", 0.9, "1 2 3 4", "test", 0.8, "4 3 2 1"],
            ["id2", "testpath2.jpg", 0.9, "1 2 3 4", "test", 0.8, "4 3 2 1"],
        ]

        ref_cols = ["image_id", "concept_class_top01", "concept_score_top01", "concept_bbox_top01", "concept_class_top02", "concept_score_top02", "concept_bbox_top02"]

        ref = pd.DataFrame(columns=ref_cols, data=ref_rows)
        np.array_equal(ref.values,res.values)
        self.assertEqual(list(ref.columns), list(res.columns))

    def test_detect_images_df_diff_len(self):
        d = TestDetectorDiffResLen()
        img_paths = [
            ("id1", "testpath1.jpg"),
            ("id2", "testpath2.jpg"),
        ]

        res = d.detect_images_df(img_paths)
        res = res.fillna('')
        ref_rows = [
            ["id1", "testpath1.jpg", 0.9, "1 2 3 4", "test", 0.8, "4 3 2 1"],
            ["id2", "testpath2.jpg", 0.9, "1 2 3 4", '', '', ''],
        ]

        ref_cols = ["image_id", "concept_class_top01", "concept_score_top01", "concept_bbox_top01", "concept_class_top02", "concept_score_top02", "concept_bbox_top02"]

        self.assertEqual(ref_rows, res.values.tolist())
        self.assertEqual(ref_cols, list(res.columns))

    def test_detect_images_auto(self):
        d = TestDetector()
        res = d.detect_images_auto()
        vc = read_vc()
        vc_l = len(vc.index)
        print(res)
        self.assertEqual(vc_l, len(res.index))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
