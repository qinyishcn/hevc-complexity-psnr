"""Offline independent checks of published features, folds, labels and metrics."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from prediction_features import feature_matrix, feature_groups
from predict_psnr import predict_portable

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'prediction'


def main():
    collection = json.loads((OUT / 'feature_collection.json').read_text())
    old_evidence = json.loads((ROOT / 'evidence/final_encodes.json').read_text())
    old_hashes = {(x['measurement']['sequence'], x['measurement']['target_kbps']):
                   x['measurement']['sha256_encoded'] for x in old_evidence}
    assert len(collection['records']) == 60
    assert {(x['sequence'], x['target_kbps']) for x in collection['records']} == set(old_hashes)
    for record in collection['records']:
        assert record['byte_identical'] and record['frames'] == 150
        assert record['sha256_original_and_logged_encoded'] == old_hashes[(record['sequence'], record['target_kbps'])]
        assert record['qp_mean_log_rounding_abs_error'] <= .011
        assert set(record['qp_frame_type_log_rounding_abs_errors']) == {'I', 'P', 'B'}
        assert max(record['qp_frame_type_log_rounding_abs_errors'].values()) <= .011
    extracted = pd.read_csv(OUT / 'encoder_features.csv')
    assert len(extracted) == 9000
    assert not extracted[['sequence', 'target_kbps', 'frame']].duplicated().any()
    assert not any(re.search('mse|psnr|distortion|energy|ssim', column, re.I) for column in extracted.columns)
    assert np.allclose(extracted.intra_cu_pct + extracted.inter_cu_pct, 100)
    assert (extracted.skip_merge_cu_pct <= extracted.inter_cu_pct + 1e-8).all()
    metrics = pd.read_csv(OUT / 'model_metrics.csv')
    per_video = pd.read_csv(OUT / 'per_video_errors.csv')
    audits = json.loads((OUT / 'fold_audit.json').read_text())
    assert len(audits) == 480
    portable = json.loads((OUT / 'ridge_models.json').read_text())
    checked_models, checked_metric_rows = 0, 0
    for level, count in [('frame', 9000), ('sequence', 60)]:
        data = pd.read_csv(OUT / f'{level}_features.csv')
        keys = ['sequence', 'target_kbps'] + (['frame'] if level == 'frame' else [])
        original = pd.read_csv(ROOT / ('frame_metrics.csv' if level == 'frame' else 'sequence_metrics.csv'))
        aligned = original.merge(data, on=keys, validate='one_to_one', suffixes=('_old', '_new'))
        assert len(aligned) == count and len(data) == count
        for column in ['psnr_y', 'mse_y', 'actual_kbps']:
            assert np.allclose(aligned[column + '_old'], aligned[column + '_new'])
        x = feature_matrix(data, level)
        assert not any(re.search('mse|psnr|distortion|energy|ssim|sequence|hash', column, re.I) for column in x.columns)
        assert all(set(columns) <= set(x.columns) for columns in feature_groups(level).values())
        for feature_set, model in portable[level].items():
            assert model['input_columns'] == feature_groups(level)[feature_set]
            assert np.isfinite(predict_portable(x, model)).all()
        oof = pd.read_csv(OUT / f'{level}_oof_predictions.csv')
        assert len(oof) == count and oof.sequence.nunique() == 20
        assert not oof[keys].duplicated().any()
        joined = data.merge(oof, on=keys, validate='one_to_one', suffixes=('_data', '_oof'))
        assert np.allclose(joined.psnr_y_data, joined.psnr_y_oof)
        for algorithm in ['ridge', 'hist_gb']:
            for feature_set in feature_groups(level):
                column = f'{algorithm}__{feature_set}'
                assert np.isfinite(oof[column]).all()
                runs = [row for row in audits if row['level'] == level and row['algorithm'] == algorithm and row['feature_set'] == feature_set]
                assert len(runs) == 20 and {row['test_video'] for row in runs} == set(data.sequence)
                for row in runs:
                    assert row['test_video'] not in row['train_videos']
                    assert len(row['train_videos']) == 19
                    assert set(row['train_videos']) | {row['test_video']} == set(data.sequence)
                    assert row['n_test'] == (450 if level == 'frame' else 3)
                    assert row['n_train'] == (8550 if level == 'frame' else 57)
                for rate in ['all', '150', '300', '600']:
                    subset = oof if rate == 'all' else oof[oof.target_kbps == int(rate)]
                    errors = subset[column] - subset.psnr_y
                    row = metrics[(metrics.level == level) & (metrics.algorithm == algorithm) &
                                  (metrics.feature_set == feature_set) & (metrics.target_kbps == rate)].iloc[0]
                    assert row.n == len(subset)
                    assert np.isclose(np.sqrt(np.mean(errors**2)), row.rmse_db)
                    assert np.isclose(np.mean(np.abs(errors)), row.mae_db)
                    assert np.isclose(1 - np.sum(errors**2) / np.sum((subset.psnr_y - subset.psnr_y.mean())**2), row.r2)
                    checked_metric_rows += 1
                for name, group in oof.groupby('sequence'):
                    row = per_video[(per_video.level == level) & (per_video.algorithm == algorithm) &
                                    (per_video.feature_set == feature_set) & (per_video.sequence == name)].iloc[0]
                    assert np.isclose(np.sqrt(np.mean((group[column] - group.psnr_y)**2)), row.rmse_db)
                checked_models += 1
    assert checked_models == 24 and checked_metric_rows == 96
    for file in ['feature_ablation', 'held_out_predictions', 'per_video_errors', 'prediction_by_bitrate']:
        for suffix in ['png', 'svg']:
            assert (OUT / 'figures' / f'{file}.{suffix}').stat().st_size > 1000
    report = (OUT / 'REPORT.md').read_text(encoding='utf-8')
    for number in ['4.041', '1.614', '3.684', '0.975', '60.1%', '73.5%']:
        assert number in report, number
    # Scan precisely the files that will be tracked, without reading local media caches.
    import subprocess
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode().splitlines()
    for name in files:
        path = ROOT / name
        assert path.suffix.lower() not in {'.mp4', '.y4m', '.yuv', '.bin', '.pdf', '.cutree', '.stats'}
        if path.suffix.lower() == '.png':
            continue
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'[A-Za-z]:[\\/]+Users[\\/]+|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}', text), name
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in OUT.rglob('*') if path.is_file() and path.name != 'validation.json'}
    result = {'status': 'passed', 'date': '2026-10-09',
        'scope': 'Offline export audit; byte-identical replay and QP log crosschecks were performed during actual feature collection.',
        'collected_encodes_byte_identical': 60, 'frame_rows': 9000, 'sequence_rows': 60,
        'original_encoder_qp_summary_and_type_checks': 60,
        'models_evaluated': checked_models, 'metric_rows_recomputed': checked_metric_rows,
        'outer_fold_train_test_source_disjoint_checks': len(audits),
        'original_psnr_mse_labels_preserved': True, 'feature_allowlist_checked': True,
        'target_error_fields_excluded_from_encoder_features': True,
        'privacy_and_excluded_media_scan': 'passed', 'artifact_sha256': hashes}
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
