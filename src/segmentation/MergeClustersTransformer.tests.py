import unittest

from src.segmentation.MergeClustersTransformer import merge_segments

class UnitTests(unittest.TestCase):

    def test_merge_no_segments(self):
        '''Merges segments which are side by side without any images between'''
        
        imgs = ["Img1", "Img2", "Img3", "Img4", "Img5", "Img6", "Img7"]

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
                [
                    "Img7",
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

        res = merge_segments(clusters, imgs, -1)

        self.assertEqual(clusters, res)

    def test_merge_segments_0(self):
        '''Merges segments which are side by side without any images between'''
        
        imgs = ["Img1", "Img2", "Img3", "Img4", "Img5", "Img6", "Img7"]

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
                [
                    "Img7",
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

        res = merge_segments(clusters, imgs, 0)

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
                    "Img7",
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

        self.assertEqual(assert_res, res)

    def test_merge_segments_1(self):
        imgs = ["Img1", "Img2", "Img3", "Img4", "Img5", "Img6"]

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

        res = merge_segments(clusters, imgs, 1)

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
                    "Img4",
                    "Img5",
                ],
            ],
        ]

        self.assertEqual(assert_res, res)

    def test_merge_segments_1_extended(self):
        imgs = ["Img1", "Img2", "Img3", "Img4", "Img5", "Img6", "Img7", "Img8", "Img9", "Img10"]

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
                    "Img8",
                ],
                [
                    "Img9",
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
                [
                    "Img7",
                ], 
                [
                    "Img10",
                ],
            ],
        ]

        res = merge_segments(clusters, imgs, 1)

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
                    "Img8",
                    "Img9",
                ],
            ],
            [
                [
                    "Img2",
                    "Img4",
                    "Img5",
                    "Img7",
                ],
                [
                    "Img10",
                ],
            ],
        ]

        self.assertEqual(assert_res, res)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
