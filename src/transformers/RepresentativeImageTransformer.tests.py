import unittest

from src.transformers.RepresentativeImageTransformer import select_first
from src.transformers.RepresentativeImageTransformer import select_last
from src.transformers.RepresentativeImageTransformer import select_all
from src.transformers.RepresentativeImageTransformer import select_first_last

class UnitTests(unittest.TestCase):
    
    def test_select_first(self):
        segments = [
            [
                "Img1",
                "Img3",
            ],
            [
                "Img6",
                "Img2",
            ],
            [
                "Img4",
                "Img5",
            ],
        ]

        res = select_first(segments)

        res_segments = [
            ["Img1"],
            ["Img6"],
            ["Img4"],
        ]

        self.assertEqual(res_segments, res)

    def test_select_last(self):
        segments = [
            [
                "Img1",
                "Img3",
            ],
            [
                "Img6",
                "Img2",
            ],
            [
                "Img4",
                "Img5",
            ],
        ]

        res = select_last(segments)

        res_segments = [
            ["Img3"],
            ["Img2"],
            ["Img5"],
        ]

        self.assertEqual(res_segments, res)

    def test_select_all(self):
        segments = [
            [
                "Img1",
                "Img3",
                "Img30",
            ],
            [
                "Img6",
                "Img2",
                "Img20",
            ],
            [
                "Img4",
                "Img5",
                "Img50",
            ],
        ]

        res = select_all(segments)

        self.assertEqual(segments, res)

    def test_select_first_last(self):
        segments = [
            [
                "Img1",
                "Img3",
                "Img30",
            ],
            [
                "Img6",
                "Img2",
                "Img20",
            ],
            [
                "Img4",
                "Img5",
                "Img50",
            ],
        ]

        res = select_first_last(segments)

        assert_res = [
            [
                "Img1",
                "Img30",
            ],
            [
                "Img6",
                "Img20",
            ],
            [
                "Img4",
                "Img50",
            ],
        ]

        self.assertEqual(assert_res, res)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
