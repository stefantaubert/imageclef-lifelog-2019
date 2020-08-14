import unittest

from src.transformers.FlatteningTransformer import flatten

class UnitTests(unittest.TestCase):

    def test_flatten(self):
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

        segments = flatten(clusters)

        assert_segments = [
            [
                "Img1",
            ],
            [
                "Img3",
            ],
            [
                "Img6",
            ],
            [
                "Img2",
            ],
            [
                "Img4",
                "Img5",
            ],
        ]

        self.assertEqual(assert_segments, segments)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
