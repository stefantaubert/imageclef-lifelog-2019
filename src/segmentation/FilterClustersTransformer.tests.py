import unittest

from src.segmentation.FilterClustersTransformer import filter_clusters

class UnitTests(unittest.TestCase):
    
    def test_filter_clusters(self):
        clusters = [
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

        res = filter_clusters(clusters, 2)

        assert_res = [
            [
                [
                    "Img4",
                    "Img5",
                ],
            ],
        ]

        self.assertEqual(assert_res, res)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
