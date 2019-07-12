import unittest

from src.submissions.converter import subm_to_df

class UnitTests(unittest.TestCase):
    
    def test_subm_to_df_has_headers(self):
        csv = subm_to_df({ })

        self.assertEqual("topic_id", csv.columns[0])
        self.assertEqual("image_id", csv.columns[1])
        self.assertEqual("confidence_score", csv.columns[2])

    def test_subm_to_df_empty(self):
        csv = subm_to_df({ })

        self.assertEqual(0, len(csv.index))

    def test_subm_to_df_normal(self):
        to_conv = {
            1: {
                'u1_20180528_1816_i00': 1.0,
                'u1_20180508_1106_i00': 0.6,
            },
            2: {
                'u1_20180528_1819_i00': 1.0,
                'u1_20180508_1106_i01': 0.8,
            },
            3: {
                'u1_20180514_1117_i07': 0.46,
                'u1_20180508_1119_i02': 0.31,
            },
        }

        csv = subm_to_df(to_conv)

        self.assertEqual(6, len(csv.index))
        self.assertEqual([1, 'u1_20180528_1816_i00', 1.0], list(csv.iloc[0]))
        self.assertEqual([1, 'u1_20180508_1106_i00', 0.6], list(csv.iloc[1]))
        self.assertEqual([2, 'u1_20180528_1819_i00', 1.0], list(csv.iloc[2]))
        self.assertEqual([2, 'u1_20180508_1106_i01', 0.8], list(csv.iloc[3]))
        self.assertEqual([3, 'u1_20180514_1117_i07', 0.46], list(csv.iloc[4]))
        self.assertEqual([3, 'u1_20180508_1119_i02', 0.31], list(csv.iloc[5]))

    def test_subm_to_df_ignore_empty(self):
        to_conv = {
            1: {
                'u1_20180528_1816_i00': 1.0,
                'u1_20180508_1106_i00': 0.6,
            },
            2: {
                'u1_20180528_1819_i00': 1.0,
                'u1_20180508_1106_i01': 0.8,
            },
            3: {
                'u1_20180514_1117_i07': 0.46,
                'u1_20180508_1119_i02': 0.31,
            },
            4: { },
            5: { }
        }

        csv = subm_to_df(to_conv)

        self.assertEqual(6, len(csv.index))
        self.assertEqual([1, 'u1_20180528_1816_i00', 1.0], list(csv.iloc[0]))
        self.assertEqual([1, 'u1_20180508_1106_i00', 0.6], list(csv.iloc[1]))
        self.assertEqual([2, 'u1_20180528_1819_i00', 1.0], list(csv.iloc[2]))
        self.assertEqual([2, 'u1_20180508_1106_i01', 0.8], list(csv.iloc[3]))
        self.assertEqual([3, 'u1_20180514_1117_i07', 0.46], list(csv.iloc[4]))
        self.assertEqual([3, 'u1_20180508_1119_i02', 0.31], list(csv.iloc[5]))

    def test_subm_to_df_unsorted_scores_are_kept(self):
        to_conv = {
            1: {
                'u1_20180508_1106_i00': 0.6,
                'u1_20180528_1816_i00': 1.0,
            },
            2: {
                'u1_20180508_1106_i01': 0.8,
                'u1_20180528_1819_i00': 1.0,
            },
            3: {
                'u1_20180508_1119_i02': 0.31,
                'u1_20180514_1117_i07': 0.46,
            },
        }

        csv = subm_to_df(to_conv)

        self.assertEqual(6, len(csv.index))
        self.assertEqual([1, 'u1_20180508_1106_i00', 0.6], list(csv.iloc[0]))
        self.assertEqual([1, 'u1_20180528_1816_i00', 1.0], list(csv.iloc[1]))
        self.assertEqual([2, 'u1_20180508_1106_i01', 0.8], list(csv.iloc[2]))
        self.assertEqual([2, 'u1_20180528_1819_i00', 1.0], list(csv.iloc[3]))
        self.assertEqual([3, 'u1_20180508_1119_i02', 0.31], list(csv.iloc[4]))
        self.assertEqual([3, 'u1_20180514_1117_i07', 0.46], list(csv.iloc[5]))

    def test_subm_to_df_keys_are_sorted(self):
        to_conv = {
            1: {
                'u1_20180508_1106_i00': 0.6,
                'u1_20180528_1816_i00': 1.0,
            },
            3: {
                'u1_20180508_1119_i02': 0.31,
                'u1_20180514_1117_i07': 0.46,
            },
            2: {
                'u1_20180508_1106_i01': 0.8,
                'u1_20180528_1819_i00': 1.0,
            },
        }

        csv = subm_to_df(to_conv)

        self.assertEqual(6, len(csv.index))
        self.assertEqual([1, 'u1_20180508_1106_i00', 0.6], list(csv.iloc[0]))
        self.assertEqual([1, 'u1_20180528_1816_i00', 1.0], list(csv.iloc[1]))
        self.assertEqual([2, 'u1_20180508_1106_i01', 0.8], list(csv.iloc[2]))
        self.assertEqual([2, 'u1_20180528_1819_i00', 1.0], list(csv.iloc[3]))
        self.assertEqual([3, 'u1_20180508_1119_i02', 0.31], list(csv.iloc[4]))
        self.assertEqual([3, 'u1_20180514_1117_i07', 0.46], list(csv.iloc[5]))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner(verbosity=2).run(suite)