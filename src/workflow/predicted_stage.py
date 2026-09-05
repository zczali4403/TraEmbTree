"""Read model-predicted stage with exact cell-index and ID validation."""
import numpy as np
import pandas as pd


def load_predicted_stage(path, cell_ids, chunk_size=500_000):
    """Accept shuffled CSV rows; require exactly one finite prediction per cell.

    idx refers to the embedding/CSR row, not the CSV row number. Every idx is
    checked against cell_id, so mismatched ordering fails rather than silently
    assigning another cell's time. true_stage and stage_id are never read.
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    n = len(cell_ids)
    values = np.empty(n, dtype=np.float64)
    seen = np.zeros(n, dtype=bool)
    with pd.read_csv(path, usecols=['idx', 'cell_id', 'predicted_stage'],
                             dtype={'idx': 'int64', 'cell_id': str, 'predicted_stage': 'float64'},
                             keep_default_na=False, chunksize=chunk_size) as reader:
        for chunk in reader:
            ids = chunk['idx'].to_numpy()
            prediction = chunk['predicted_stage'].to_numpy()
            if np.any(ids < 0) or np.any(ids >= n):
                raise ValueError('predicted-stage idx is outside the embedding/CSR row range')
            if len(np.unique(ids)) != len(ids) or seen[ids].any():
                raise ValueError('duplicate idx in predicted-stage CSV')
            expected = np.asarray(cell_ids[ids]).astype(str)
            if not np.array_equal(expected, chunk['cell_id'].to_numpy()):
                raise ValueError('predicted-stage cell_id does not match CSR cell_ids at idx')
            if not np.isfinite(prediction).all():
                raise ValueError('predicted_stage must contain finite values for every cell')
            values[ids] = prediction
            seen[ids] = True
    if not seen.all():
        raise ValueError(f'predicted-stage CSV is missing {np.count_nonzero(~seen)} cells')
    return values
