import unittest

from src.segmentation.ClusterSplitterTransformer import build_clusters

class UnitTests(unittest.TestCase):

    def test_build_clusters(self):
        imgs = ["Img1", "Img2", "Img3", "Img4", "Img5", "Img6"]
        segments = [1, 3, 2, 3, 3, 2]

        assert_res = [
            [
                [
                    "Img1",
                ],
            ],
            [
                [
                    "Img3",
                ],
                [
                    "Img6",
                ],
            ],
            [
                [
                    "Img2",
                ],
                [
                    "Img4",
                    "Img5",
                ],
            ],
        ]

        res = build_clusters(imgs, segments)

        self.assertEqual(assert_res, res)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
