"""Re-log final pass2 in an isolated cache and require identical encoded bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from prediction_features import read_encoder_csv

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_brightness(path):
    rows = []
    with path.open('rb') as stream:
        stream.readline()
        for frame in range(150):
            assert stream.readline() == b'FRAME\n'
            data = stream.read(352 * 288 * 3 // 2)
            y = np.frombuffer(data, dtype=np.uint8, count=352 * 288)
            rows.append({'frame': frame, 'source_luma_mean': y.mean(), 'source_luma_std': y.std()})
    return pd.DataFrame(rows)


def collect_one(item, experiment_root):
    measurement = item['measurement']
    name, rate = measurement['sequence'], measurement['target_kbps']
    original = experiment_root / 'runs' / name / str(rate)
    folder = ROOT / 'prediction_cache' / name / str(rate)
    folder.mkdir(parents=True, exist_ok=True)
    command = next(value for key, value in item['final_commands'].items() if key.endswith('pass2'))
    command = [str(experiment_root / value[2:]) if value.startswith(('./', '.\\')) else value for value in command]
    command[-1] = str(folder / 'encoded.mp4')
    command[command.index('-x265-params') + 1] += ':csv=encoder.csv:csv-log-level=2'
    if not (folder / 'encoded.mp4').exists() or not (folder / 'encoder.csv').exists():
        for suffix in ['', '.cutree']:
            shutil.copyfile(original / ('rate.stats' + suffix), folder / ('rate.stats' + suffix))
        # Logging appends; never mix runs if recovering an interrupted attempt.
        (folder / 'encoder.csv').unlink(missing_ok=True)
        start = time.perf_counter()
        with (folder / 'encode.log').open('w', encoding='utf-8') as stream:
            subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT, check=True)
        (folder / 'elapsed.json').write_text(json.dumps({'pass2_wall_seconds': time.perf_counter() - start}))
    observed = sha256(folder / 'encoded.mp4')
    if observed != measurement['sha256_encoded']:
        raise RuntimeError(f'{name}/{rate}: logged replay changed encoded bytes; labels cannot be reused')
    features = read_encoder_csv(folder / 'encoder.csv')
    assert len(features) == 150
    attempt = measurement['calibration_attempts'][-1]['attempt']
    original_log = (original / f'attempt{attempt}_pass2.log').read_text(encoding='utf-8')
    summary_qp = float(re.findall(r'encoded 150 frames[^\n]*Avg QP:([\d.]+)', original_log)[-1])
    qp_error = abs(features.qp.mean() - summary_qp)
    assert qp_error <= .011, (name, rate, qp_error)
    qp_type_errors = {}
    for kind, value in re.findall(r'frame ([IPB]):\s+\d+, Avg QP:([\d.]+)', original_log):
        qp_type_errors[kind] = abs(features.loc[features.pict_type == kind, 'qp'].mean() - float(value))
        assert qp_type_errors[kind] <= .011, (name, rate, kind, qp_type_errors[kind])
    features['sequence'], features['target_kbps'] = name, rate
    elapsed = json.loads((folder / 'elapsed.json').read_text()) if (folder / 'elapsed.json').exists() else {}
    normalized = [value.replace(str(experiment_root), '<SOURCE_ROOT>').replace(str(ROOT / 'prediction_cache'), '<CACHE_ROOT>') for value in command]
    record = {'sequence': name, 'target_kbps': rate, 'sha256_original_and_logged_encoded': observed,
              'byte_identical': True, 'frames': len(features), 'raw_encoder_csv_sha256': sha256(folder / 'encoder.csv'),
              'first_pass_stats_sha256': sha256(original / 'rate.stats'), 'final_command': normalized,
              'original_encoder_summary_qp': summary_qp, 'qp_mean_log_rounding_abs_error': qp_error,
              'qp_frame_type_log_rounding_abs_errors': qp_type_errors, **elapsed}
    return features, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    source = args.experiment_root.resolve()
    output = ROOT / 'prediction'
    output.mkdir(exist_ok=True)
    evidence = json.loads((ROOT / 'evidence/final_encodes.json').read_text(encoding='utf-8'))
    manifests = json.loads((ROOT / 'dataset_manifest.json').read_text(encoding='utf-8'))
    brightness = []
    for manifest in manifests:
        path = source / 'data' / manifest['sequence'] / 'reference.y4m'
        assert sha256(path) == manifest['sha256_reference']
        frame = source_brightness(path)
        frame['sequence'] = manifest['sequence']
        brightness.append(frame)
    results, records = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect_one, item, source) for item in evidence]
        for future in as_completed(futures):
            features, record = future.result()
            results.append(features)
            records.append(record)
            print(f"LOGGED {record['sequence']} {record['target_kbps']} byte-identical", flush=True)
    features = pd.concat(results).merge(pd.concat(brightness), on=['sequence', 'frame'], validate='many_to_one')
    features = features.sort_values(['sequence', 'target_kbps', 'frame']).reset_index(drop=True)
    features.to_csv(output / 'encoder_features.csv', index=False)
    original = pd.read_csv(ROOT / 'frame_metrics.csv')
    merged = original.merge(features, on=['sequence', 'target_kbps', 'frame'], validate='one_to_one', suffixes=('', '_logged'))
    assert len(merged) == 9000 and (merged.pict_type == merged.pict_type_logged).all()
    merged = merged.drop(columns='pict_type_logged')
    merged.to_csv(output / 'frame_features.csv', index=False)
    aggregated = []
    for (name, rate), group in merged.groupby(['sequence', 'target_kbps']):
        aggregated.append({'sequence': name, 'target_kbps': rate,
            'qp_mean': group.qp.mean(), 'qp_std': group.qp.std(ddof=0),
            'qp_I_mean': group.loc[group.pict_type == 'I', 'qp'].mean(),
            'qp_P_mean': group.loc[group.pict_type == 'P', 'qp'].mean(),
            'qp_B_mean': group.loc[group.pict_type == 'B', 'qp'].mean(),
            'intra_cu_pct_mean': group.intra_cu_pct.mean(), 'inter_cu_pct_mean': group.inter_cu_pct.mean(),
            'skip_merge_cu_pct_mean': group.skip_merge_cu_pct.mean(),
            'packet_bpp_std': (group.packet_bytes * 8 / (352 * 288)).std(ddof=0),
            'source_luma_mean': group.source_luma_mean.mean(), 'source_luma_std_mean': group.source_luma_std.mean(),
            'I_frame_fraction': (group.pict_type == 'I').mean(),
            'P_frame_fraction': (group.pict_type == 'P').mean(),
            'B_frame_fraction': (group.pict_type == 'B').mean()})
    sequences = pd.read_csv(ROOT / 'sequence_metrics.csv').merge(pd.DataFrame(aggregated),
        on=['sequence', 'target_kbps'], validate='one_to_one')
    sequences.to_csv(output / 'sequence_features.csv', index=False)
    record = {'status': 'passed', 'encodes': 60, 'frames': 9000,
              'all_logged_encodes_byte_identical': True, 'source_references_sha256_checked': 20,
              'target_leakage_columns_excluded_from_encoder_features':
                  ['PSNR', 'SSIM', 'Avg Luma Distortion', 'Avg Chroma Distortion', 'Avg psyEnergy', 'Avg Residual Energy'],
              'cu_percentage_definition': 'First global CU count block; not pixel-area fractions. Skip+Merge branches count toward inter.',
              'records': sorted(records, key=lambda x: (x['sequence'], x['target_kbps']))}
    (output / 'feature_collection.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Collected and aligned 9000 frames across 60 byte-identical logged encodes')


if __name__ == '__main__':
    main()
