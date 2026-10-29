"""Check preserved inputs and numerical reproduction against frozen results."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    for relative, entry in manifest['input_files'].items():
        actual = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise ValueError(f'Input hash mismatch: {relative}')
    frozen = json.loads((root/'expected.json').read_text(encoding='utf-8'))
    rows = 0
    for filename, records in frozen['tables'].items():
        actual = pd.read_csv(args.output/filename)
        expected = pd.DataFrame(records)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-10)
        rows += len(actual)
    report = json.loads((args.output/'revision_sensitivity.json').read_text(encoding='utf-8'))
    references = {
        'primary': (-1.4266536670, 0.55224609375),
        'final': (-0.9608868404, 0.56298828125),
    }
    for layer, (median, p) in references.items():
        record = report['sign_flip'][layer]
        np.testing.assert_allclose(record['median_excess_pp'], median, rtol=0, atol=1e-9)
        np.testing.assert_allclose(record['independent_cell_p'], p, rtol=0, atol=1e-12)
        if record['dataset_block_p'] != 1.0:
            raise ValueError(f'{layer}: unexpected block p')
        window = report['window_audit'][layer]
        if window != {'fits': 360, 'bins_per_fit': 5, 'undefined_bins': 0,
                      'supplied_minimum_matches_all_fits': True}:
            raise ValueError(f'{layer}: window accounting mismatch')
    np.testing.assert_allclose(report['numeric_rho_sensitivity']['rho'], 0.7333333333333333,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(report['numeric_rho_sensitivity']['bca_ci95'],
                               [0.1794871794871795, 1.0], rtol=0, atol=1e-10)
    print(f'PASS: {len(frozen["tables"])} reference tables ({rows} rows), input hashes, '
          'sign-flip statistics, BCa range, and window accounting.')


if __name__ == '__main__':
    main()
