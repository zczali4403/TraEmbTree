from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json

import numpy as np
import pandas as pd

UTILITY = Path(__file__).resolve().parents[1] / 'src' / 'utilities'
sys.path.insert(0, str(UTILITY))
from evaluate_microcell_purity import evaluate, summary


def fixture():
    micro = pd.DataFrame({'microcell_id': [0, 1, 2], 'n_cells': [10, 20, 1], 'lineage': ['L1', 'L2', 'L2']})
    composition = pd.DataFrame({'microcell_id': [0, 0, 1, 1], 'celltype': ['A', 'Unknown', 'B', 'C'], 'cell_count': [6, 2, 10, 10]})
    return micro, composition


class PurityTests(unittest.TestCase):
    def test_denominators_and_missing_labels(self):
        frame = evaluate(*fixture())
        self.assertAlmostEqual(frame.iloc[0].purity, 1.)
        self.assertAlmostEqual(frame.iloc[0].purity_including_unknown, .75)
        self.assertAlmostEqual(frame.iloc[0].dominant_fraction_all_cells, .6)
        self.assertTrue(np.isnan(frame.iloc[2].purity))
        report = summary(frame, 5)
        self.assertAlmostEqual(report['mean_purity'], .75)
        self.assertAlmostEqual(report['known_cell_weighted_purity'], 16/26)
        self.assertAlmostEqual(report['dominant_fraction_all_cells'], 16/31)
        self.assertEqual(report['unknown_cells'], 2)
        self.assertEqual(report['unmatched_cells'], 3)
        self.assertAlmostEqual(report['purity_ge_0.9_all_cell_coverage'], 10/31)

    def test_invalid_counts_and_empty_annotations(self):
        micro, comp = fixture()
        for bad in (comp.assign(cell_count=[60, 2, 10, 10]), comp.assign(cell_count=[.5, 2, 10, 10]),
                    comp.assign(microcell_id=[99, 0, 1, 1])):
            with self.assertRaises(ValueError):
                evaluate(micro, bad)
        report = summary(evaluate(micro, comp.iloc[:0]), 50)
        self.assertIsNone(report['mean_purity'])
        self.assertEqual(report['unmatched_cells'], 31)
        json.dumps(report, allow_nan=False)

    def test_cli_figures_with_and_without_known_labels(self):
        micro, comp = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            micro.to_parquet(root / 'microcells.parquet', index=False)
            for run, composition in enumerate((comp, comp.assign(celltype='Unknown'))):
                composition.to_parquet(root / 'microcell_celltype_composition.parquet', index=False)
                output = root / f'evaluation{run}'
                result = subprocess.run([sys.executable, str(UTILITY / 'evaluate_microcell_purity.py'),
                                         '--input-dir', str(root), '--output-dir', str(output), '--dpi', '50'],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(len(list(output.glob('*.png'))), 5)
                self.assertEqual(len(list(output.glob('*.pdf'))), 5)
                self.assertEqual(len(pd.read_csv(output / 'per_microcell.csv')), 3)
                report = json.loads((output / 'summary.json').read_text())
                if run:
                    self.assertIsNone(report['mean_purity'])


if __name__ == '__main__':
    unittest.main()
