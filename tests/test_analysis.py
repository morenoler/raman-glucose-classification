import unittest

import numpy as np
import pandas as pd

from src.train import scores, snv, well_ids


class AnalysisTests(unittest.TestCase):
    def test_snv_normalizes_each_spectrum_independently(self):
        values = snv(np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]))
        np.testing.assert_allclose(values.mean(axis=1), [0, 0], atol=1e-12)
        np.testing.assert_allclose(values.std(axis=1), [1, 1], atol=1e-12)

    def test_well_id_groups_repeated_measurements(self):
        metadata = pd.DataFrame({"plate": [1, 1, 1], "row": ["A", "A", "B"], "col": [2, 2, 2]})
        ids = well_ids(metadata)
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[0], ids[2])

    def test_balanced_accuracy_exposes_majority_class_baseline(self):
        actual = np.array([0, 0, 1, 1, 1, 1])
        predicted = np.ones(len(actual), dtype=int)
        result = scores(actual, predicted, predicted.astype(float))
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["confusion_tn_fp_fn_tp"], [0, 2, 0, 4])


if __name__ == "__main__":
    unittest.main()
