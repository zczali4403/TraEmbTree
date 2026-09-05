import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import faiss
import numpy as np
import pandas as pd

WORKFLOW = Path(__file__).resolve().parents[1] / 'src' / 'workflow'
sys.path.insert(0, str(WORKFLOW))
from cohesive_microcells import cohesive_partition

faiss.omp_set_num_threads(1)


class CohesiveTests(unittest.TestCase):
    def test_separated_rare_group_is_not_forced_into_large_group(self):
        rng = np.random.default_rng(7)
        x = np.concatenate([np.array([1., 0., 0.]) + rng.normal(0, .002, (60, 3)),
                            np.array([0., 1., 0.]) + rng.normal(0, .002, (6, 3))])
        labels = cohesive_partition(x, 20, 10, 25, k=5)
        self.assertFalse(set(labels[:60]) & set(labels[60:]))
        self.assertLessEqual(np.bincount(labels).max(), 25)
        np.testing.assert_array_equal(labels, cohesive_partition(x, 20, 10, 25, k=5))

    def test_identical_vectors_and_small_inputs(self):
        for n in (0, 1, 2, 31):
            x = np.tile([1., 0.], (n, 1))
            for mutual in (False, True):
                labels = cohesive_partition(x, 7, 3, 10, k=5, mutual=mutual)
                self.assertEqual(len(labels), n)
                if n:
                    self.assertGreaterEqual(labels.min(), 0)
                    self.assertLessEqual(np.bincount(labels).max(), 10)
                    np.testing.assert_array_equal(np.unique(labels), np.arange(labels.max() + 1))

    def test_invalid_vectors_and_sizes(self):
        for x in (np.zeros((2, 3)), np.array([[np.nan, 1.]])):
            with self.assertRaises(ValueError):
                cohesive_partition(x, 5, 2, 10)
        with self.assertRaises(ValueError):
            cohesive_partition(np.ones((2, 3)), 5, 6, 10)

    def test_pipeline_and_label_independence(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.concatenate([np.array([1., 0., 0.]) + rng.normal(0, .01, (35, 3)),
                                np.array([0., 1., 0.]) + rng.normal(0, .01, (35, 3))]).astype('float32')
            np.save(root / 'embeddings.npy', x)
            np.save(root / 'lineage_ids.npy', np.zeros(70, dtype='int32'))
            np.save(root / 'stage_ids.npy', np.zeros(70, dtype='int32'))
            np.save(root / 'cell_ids.npy', np.array([f'c{i}' for i in range(70)]))
            (root / 'metadata.json').write_text(json.dumps({'lineages': ['L'], 'stages': [2.]}))
            assignments = []
            for run, types in enumerate((['A'] * 35 + ['B'] * 35, ['changed'] * 70, ['A'] * 35 + ['B'] * 35, ['changed'] * 70)):
                metadata = pd.DataFrame({'cell': [f'c{i}' for i in range(70)], 'celltype': types})
                metadata.to_csv(root / 'metadata.csv', index=False)
                output = root / f'output{run}'
                cmd = [sys.executable, str(WORKFLOW / 'select_landmark_nodes_mc2_lineage_stagebin_marginal.py'),
                       '--embeddings', str(root / 'embeddings.npy'), '--csr-dir', str(root),
                       '--metadata', str(root / 'metadata.csv'), '--output-dir', str(output),
                       '--target-nodes', '2', '--microcell-size', '10', '--microcell-min-size', '3',
                       '--microcell-max-size', '15', '--pile-size', '40', '--knn', '5', '--faiss-threads', '1']
                if run >= 2:
                    # Predicted mode works without original stage files or metadata.
                    if run == 2:
                        (root / 'stage_ids.npy').unlink()
                        (root / 'metadata.json').write_text(json.dumps({'lineages': ['L']}))
                    pd.DataFrame({'idx': np.arange(70), 'cell_id': [f'c{i}' for i in range(70)],
                                  'predicted_stage': [.25] * 35 + [2.75] * 35}).iloc[::-1].to_csv(root / 'pred.csv', index=False)
                    cmd += ['--predicted-stage', str(root / 'pred.csv')]
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                labels = np.load