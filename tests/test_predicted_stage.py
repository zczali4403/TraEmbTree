from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'workflow'))
from predicted_stage import load_predicted_stage


class PredictedStageTests(unittest.TestCase):
    def test_shuffled_rows_and_unused_original_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pred.csv'
            pd.DataFrame({'idx': [2, 0, 1], 'cell_id': ['c', 'a', 'b'],
                          'predicted_stage': [3.5, -0.2, 1.7],
                          'true_stage': ['unused'] * 3}).to_csv(path, index=False)
            np.testing.assert_allclose(load_predicted_stage(path, np.array(['a', 'b', 'c']), 1),
                                       [-.2, 1.7, 3.5])

    def test_invalid_alignment_duplicates_missing_and_nonfinite(self):
        base = pd.DataFrame({'idx': [0, 1], 'cell_id': ['a', 'b'], 'predicted_stage': [1., 2.]})
        cases = [base.assign(cell_id=['b', 'a']), base.assign(idx=[0, 0]),
                 base.iloc[:1], base.assign(predicted_stage=[1., np.inf]),
                 base.assign(idx=[0, 2]), base.assign(idx=[0, -1])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pred.csv'
            for frame in cases:
                frame.to_csv(path, index=False)
                with self.subTest(data=frame.to_dict()), self.assertRaises(ValueError):
                    load_predicted_stage(path, np.array(['a', 'b']), 1)


if __name__ == '__main__':
    unittest.main()
