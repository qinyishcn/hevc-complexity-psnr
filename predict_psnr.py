"""Grouped out-of-video PSNR prediction with nested ridge selection and ablations."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from prediction_features import feature_groups, feature_matrix

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'prediction'
SEED = 20261009
ALPHAS = [.1, 1., 10., 100.]


def grouped_splits(groups, n_splits=None):
    groups = np.asarray(groups)
    splitter = LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits=n_splits)
    for train, test in splitter.split(np.zeros(len(groups)), groups=groups):
        assert not set(groups[train]) & set(groups[test])
        yield train, test


def fit_model(name, x, y, groups, level):
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    if name == 'ridge':
        model = make_pipeline(imputer, StandardScaler(), Ridge())
        search = GridSearchCV(model, {'ridge__alpha': ALPHAS}, scoring='neg_mean_squared_error',
                              cv=list(grouped_splits(groups, n_splits=3)), n_jobs=1)
        search.fit(x, y)
        return search.best_estimator_, float(search.best_params_['ridge__alpha'])
    model = make_pipeline(imputer, HistGradientBoostingRegressor(
        max_iter=150, learning_rate=.06, max_leaf_nodes=7, l2_regularization=10.,
        min_samples_leaf=30 if level == 'frame' else 5, early_stopping=False, random_state=SEED))
    model.fit(x, y)
    return model, None


def scores(actual, predicted):
    return {'rmse_db': float(np.sqrt(mean_squared_error(actual, predicted))),
            'mae_db': float(mean_absolute_error(actual, predicted)),
            'r2': float(r2_score(actual, predicted)),
            'bias_db': float(np.mean(predicted - actual)), 'n': len(actual)}


def paired_comparisons(oof, level):
    rows = []
    rng = np.random.default_rng(SEED)
    names = sorted(oof.sequence.unique())
    bootstrap = rng.integers(0, len(names), (10000, len(names)))
    for algorithm in ['ridge', 'hist_gb']:
        for before, after in [('si_ti', 'si_ti_qp'), ('si_ti', 'si_ti_qp_cu'),
                              ('si_ti_qp', 'si_ti_qp_cu'), ('si_ti_qp_cu', 'all_easy')]:
            columns = [f'{algorithm}__{before}', f'{algorithm}__{after}']
            by_video = []
            for name in names:
                group = oof[oof.sequence == name]
                by_video.append([np.mean((group[column] - group.psnr_y)**2) for column in columns])
            errors = np.asarray(by_video)
            resampled = errors[bootstrap].mean(axis=1)
            delta = np.sqrt(resampled[:, 0]) - np.sqrt(resampled[:, 1])
            before_rmse, after_rmse = np.sqrt(errors.mean(axis=0))
            rows.append({'level': level, 'algorithm': algorithm, 'before': before, 'after': after,
                         'before_rmse_db': before_rmse, 'after_rmse_db': after_rmse,
                         'rmse_reduction_db': before_rmse - after_rmse,
                         'rmse_reduction_pct': 100 * (before_rmse - after_rmse) / before_rmse,
                         'paired_video_bootstrap_low_db': np.quantile(delta, .025),
                         'paired_video_bootstrap_high_db': np.quantile(delta, .975),
                         'videos_improved_of_20': int((errors[:, 1] < errors[:, 0]).sum())})
    return rows


def evaluate(level):
    data = pd.read_csv(OUTPUT / f'{level}_features.csv').sort_values(
        ['sequence', 'target_kbps'] + (['frame'] if level == 'frame' else [])).reset_index(drop=True)
    x, y, groups = feature_matrix(data, level), data.psnr_y.to_numpy(), data.sequence.to_numpy()
    key_columns = ['sequence', 'target_kbps'] + (['frame', 'pict_type'] if level == 'frame' else [])
    oof = data[key_columns + ['psnr_y']].copy()
    metrics, folds, audit, timings, exported = [], [], [], [], {}
    schemas = feature_groups(level)
    splits = list(grouped_splits(groups))
    for algorithm in ['ridge', 'hist_gb']:
        for feature_set, columns in schemas.items():
            prediction = np.full(len(data), np.nan)
            start = time.perf_counter()
            for train, test in splits:
                fitted, alpha = fit_model(algorithm, x.iloc[train][columns], y[train], groups[train], level)
                prediction[test] = fitted.predict(x.iloc[test][columns])
                held = str(groups[test][0])
                audit.append({'level': level, 'algorithm': algorithm, 'feature_set': feature_set,
                              'test_video': held, 'train_videos': sorted(set(groups[train])),
                              'n_train': len(train), 'n_test': len(test), 'selected_alpha': alpha})
                folds.append({'level': level, 'algorithm': algorithm, 'feature_set': feature_set,
                              'sequence': held, **scores(y[test], prediction[test])})
            assert np.isfinite(prediction).all()
            oof[f'{algorithm}__{feature_set}'] = prediction
            metrics.append({'level': level, 'algorithm': algorithm, 'feature_set': feature_set,
                            'target_kbps': 'all', **scores(y, prediction)})
            for rate in [150, 300, 600]:
                mask = data.target_kbps == rate
                metrics.append({'level': level, 'algorithm': algorithm, 'feature_set': feature_set,
                                'target_kbps': str(rate), **scores(y[mask], prediction[mask])})
            elapsed = time.perf_counter() - start
            timings.append({'level': level, 'algorithm': algorithm, 'feature_set': feature_set,
                            'outer_20_folds_wall_seconds': elapsed})
            print(f'{level} {algorithm} {feature_set}: RMSE={metrics[-4]["rmse_db"]:.4f}, {elapsed:.1f}s', flush=True)
    # Fit exportable ridge models on all data, for subsequent unseen inputs only.
    for feature_set in ['si_ti', 'si_ti_qp_cu', 'all_easy']:
        columns = schemas[feature_set]
        model, alpha = fit_model('ridge', x[columns], y, groups, level)
        imputer, scaler, ridge = model.steps[0][1], model.steps[1][1], model.steps[2][1]
        exported[feature_set] = {'input_columns': columns, 'median': imputer.statistics_.tolist(),
            'missing_indicator_indices': imputer.indicator_.features_.tolist(),
            'scale_mean': scaler.mean_.tolist(), 'scale_std': scaler.scale_.tolist(),
            'coefficient': ridge.coef_.tolist(), 'intercept': float(ridge.intercept_), 'alpha': alpha}
        # Check the portable JSON formula agrees with sklearn before publishing.
        portable = predict_portable(x[columns], exported[feature_set])
        assert np.allclose(portable, model.predict(x[columns]), atol=1e-10)
        start = time.perf_counter()
        for _ in range(100):
            predict_portable(x[columns], exported[feature_set])
        timings.append({'level': level, 'algorithm': 'portable_ridge', 'feature_set': feature_set,
                        'batch_inference_microseconds_per_row': (time.perf_counter() - start) * 1e6 / (100 * len(x)),
                        'note': 'Batch CPU prediction including pandas-to-numpy; excludes video encode, SI/TI, and collection.'})
    oof.to_csv(OUTPUT / f'{level}_oof_predictions.csv', index=False)
    return metrics, folds, paired_comparisons(oof, level), audit, timings, exported


def predict_portable(x, model):
    values = x[model['input_columns']].to_numpy(dtype=float)
    missing = np.isnan(values)
    values = np.where(missing, np.asarray(model['median']), values)
    if model['missing_indicator_indices']:
        values = np.column_stack([values, missing[:, model['missing_indicator_indices']].astype(float)])
    values = (values - np.asarray(model['scale_mean'])) / np.asarray(model['scale_std'])
    return values @ np.asarray(model['coefficient']) + model['intercept']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=Path, help='Predict from a frame_features/sequence_features-format CSV')
    parser.add_argument('--level', choices=['frame', 'sequence'], default='frame')
    parser.add_argument('--feature-set', choices=['si_ti', 'si_ti_qp_cu', 'all_easy'], default='si_ti_qp_cu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.predict:
            raw = pd.read_csv(args.predict)
            model = json.loads((OUTPUT / 'ridge_models.json').read_text())[args.level][args.feature_set]
            result = raw[['sequence', 'target_kbps'] + (['frame'] if args.level == 'frame' else [])].copy()
            result['predicted_psnr_y'] = predict_portable(feature_matrix(raw, args.level, model['input_columns']), model)
            result.to_csv(args.output or OUTPUT / 'inference_predictions.csv', index=False)
            return
        all_metrics, all_folds, comparisons, audits, timings, models = [], [], [], [], [], {}
        for level in ['frame', 'sequence']:
            metrics, folds, paired, audit, timing, fitted = evaluate(level)
            all_metrics.extend(metrics); all_folds.extend(folds); comparisons.extend(paired)
            audits.extend(audit); timings.extend(timing); models[level] = fitted
        pd.DataFrame(all_metrics).to_csv(OUTPUT / 'model_metrics.csv', index=False)
        pd.DataFrame(all_folds).to_csv(OUTPUT / 'per_video_errors.csv', index=False)
        pd.DataFrame(comparisons).to_csv(OUTPUT / 'paired_comparisons.csv', index=False)
        pd.DataFrame(timings).to_csv(OUTPUT / 'timings.csv', index=False)
        (OUTPUT / 'fold_audit.json').write_text(json.dumps(audits, indent=2) + '\n')
        (OUTPUT / 'ridge_models.json').write_text(json.dumps(models, indent=2) + '\n')
        configuration = {'seed': SEED, 'python': platform.python_version(), 'sklearn': sklearn.__version__,
            'outer': '20 LeaveOneGroupOut folds by source video, all three rates and all frames held together',
            'ridge_inner': '3 GroupKFold folds over outer-training videos only', 'ridge_alphas': ALPHAS,
            'nonlinear': {'name': 'HistGradientBoostingRegressor', 'max_iter': 150, 'learning_rate': .06,
                          'max_leaf_nodes': 7, 'l2_regularization': 10, 'early_stopping': False,
                          'min_samples_leaf_frame': 30, 'min_samples_leaf_sequence': 5},
            'feature_groups': {level: feature_groups(level) for level in ['frame', 'sequence']},
            'primary_comparison': 'ridge si_ti vs ridge si_ti_qp_cu, frame-level OOF PSNR-Y',
            'bootstrap': '10000 paired whole-video resamples of fixed OOF errors; conditional descriptive CI, no model refitting',
            'exported_models': 'Full-cohort ridge fit for subsequent unseen inputs; never used as evaluation predictions',
            'training_thread_limit': 1}
        (OUTPUT / 'model_configuration.json').write_text(json.dumps(configuration, indent=2) + '\n')


if __name__ == '__main__':
    main()
