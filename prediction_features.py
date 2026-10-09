"""Parse x265 frame statistics and define an explicit non-target feature schema."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def read_encoder_csv(path, gop=60):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.reader(stream)
        header = [x.strip() for x in next(rows)]
        # Later PU columns duplicate CU names: use only the first, global block.
        start = header.index('Intra 64x64 DC')
        end = header.index('Avg Luma Distortion')
        index = {key: header.index(key) for key in ['Encode Order', 'Type', 'POC', 'QP', 'Bits']}
        result, base, last_order = [], -gop, -1
        for row in rows:
            if not row or len(row) < end:
                continue
            try:
                order = int(row[index['Encode Order']])
            except ValueError:
                continue  # x265 may append a summary header.
            if order != last_order + 1:
                raise ValueError('CSV must contain one contiguous encoder run')
            last_order = order
            kind = row[index['Type']].strip()[0].upper()
            poc = int(row[index['POC']])
            if kind == 'I':
                if poc != 0 or order % gop:
                    raise ValueError('Expected fixed closed GOP with IDR POC reset')
                base += gop
            if base < 0 or not 0 <= poc < gop:
                raise ValueError('Invalid GOP/POC alignment')
            totals = {'intra': 0., 'inter': 0., 'skip': 0., 'merge': 0.}
            for name, value in zip(header[start:end], row[start:end]):
                group = 'intra' if name == '4x4' else name.split()[0].lower()
                if group in totals:
                    totals[group] += float(value.strip().rstrip('%'))
            total = sum(totals.values())
            if abs(total - 100) > .16:
                raise ValueError(f'Global CU mode fractions do not sum to 100: {total}')
            # Normalize the small error introduced by printed 2-decimal percentages.
            totals = {key: value * 100 / total for key, value in totals.items()}
            result.append({'frame': base + poc, 'encode_order': order, 'local_poc': poc,
                           'pict_type': kind, 'qp': float(row[index['QP']]),
                           'encoder_bits': int(row[index['Bits']]),
                           'intra_cu_pct': totals['intra'],
                           'inter_cu_pct': totals['inter'] + totals['skip'] + totals['merge'],
                           'inter_nonskip_cu_pct': totals['inter'],
                           'skip_merge_cu_pct': totals['skip'] + totals['merge'],
                           'cu_rounding_total_pct': total})
    result = pd.DataFrame(result).sort_values('frame').reset_index(drop=True)
    if result.frame.tolist() != list(range(len(result))):
        raise ValueError('Missing or duplicate display frames')
    if not result.qp.between(0, 69).all():
        raise ValueError('Invalid x265 average QP')
    return result


def feature_matrix(data, level, columns=None):
    """Never infer predictors by dropping a target: construct only allowed fields."""
    fields = {'log2_target_kbps': lambda: np.log2(data.target_kbps)}
    if level == 'frame':
        fields.update({'is_I': lambda: (data.pict_type == 'I').astype(float),
            'is_P': lambda: (data.pict_type == 'P').astype(float),
            'gop_position': lambda: data.frame % 60 / 59,
            'si_log': lambda: np.log1p(data.si_2023), 'ti_log': lambda: np.log1p(data.ti_2023),
            'qp': lambda: data.qp, 'intra_cu_pct': lambda: data.intra_cu_pct,
            'skip_merge_cu_pct': lambda: data.skip_merge_cu_pct,
            'log1p_packet_bpp': lambda: np.log1p(data.packet_bytes * 8 / (352 * 288)),
            'source_luma_mean': lambda: data.source_luma_mean,
            'source_luma_std': lambda: data.source_luma_std})
    elif level == 'sequence':
        fields.update({'si_log': lambda: np.log1p(data.si_2023_mean),
                       'ti_log': lambda: np.log1p(data.ti_2023_mean)})
        for column in ['qp_mean', 'qp_std', 'intra_cu_pct_mean', 'skip_merge_cu_pct_mean',
                       'packet_bpp_std', 'source_luma_mean', 'source_luma_std_mean']:
            fields[column] = lambda column=column: data[column]
    else:
        raise ValueError(level)
    chosen = list(fields) if columns is None else columns
    if set(chosen) - set(fields):
        raise ValueError('Predictors must belong to the explicit feature allowlist')
    return pd.DataFrame({column: fields[column]() for column in chosen}, index=data.index)


def feature_groups(level):
    common = ['log2_target_kbps']
    if level == 'frame':
        common += ['is_I', 'is_P', 'gop_position']
        qp = ['qp']
        cu = ['intra_cu_pct', 'skip_merge_cu_pct']
        extra = ['log1p_packet_bpp', 'source_luma_mean', 'source_luma_std']
    else:
        qp = ['qp_mean', 'qp_std']
        cu = ['intra_cu_pct_mean', 'skip_merge_cu_pct_mean']
        extra = ['packet_bpp_std', 'source_luma_mean', 'source_luma_std_mean']
    baseline = common + ['si_log', 'ti_log']
    return {'rate_type': common, 'si_ti': baseline, 'qp_only': common + qp,
            'si_ti_qp': baseline + qp, 'si_ti_qp_cu': baseline + qp + cu,
            'all_easy': baseline + qp + cu + extra}
