"""Add current P.910 SDR SI/TI without changing already measured encodes."""
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT, N, W, H, FRAME_BYTES, SEQUENCES
from metrics import pq_luma, spatial_information, temporal_information


def main():
    lookup = pq_luma(np.arange(256))
    lookup_full = pq_luma(np.arange(256), color_range='full')
    rows = []
    for name in SEQUENCES:
        folder = ROOT / 'data' / name
        old = pd.read_csv(folder / 'complexity.csv')
        si, ti, si_full, ti_full, outside = [], [], [], [], 0
        previous = None
        previous_full = None
        with (folder / 'reference.y4m').open('rb') as inp:
            inp.readline()
            for i in range(N):
                assert inp.readline() == b'FRAME\n'
                raw = np.frombuffer(inp.read(FRAME_BYTES), dtype=np.uint8)[:W*H].reshape(H, W)
                outside += int(np.count_nonzero((raw < 16) | (raw > 235)))
                y = lookup[raw]
                si.append(spatial_information(y))
                ti.append(temporal_information(y, previous) if i else np.nan)
                previous = y
                full = lookup_full[raw]
                si_full.append(spatial_information(full))
                ti_full.append(temporal_information(full, previous_full) if i else np.nan)
                previous_full = full
        old['si_2023'] = si; old['ti_2023'] = ti
        old['si_2023_full'] = si_full; old['ti_2023_full'] = ti_full
        old.to_csv(folder / 'complexity.csv', index=False)
        rows.append({'sequence': name, 'si_2023_mean': float(np.mean(si)),
                     'ti_2023_mean': float(np.nanmean(ti)), 'si_2023_max': float(np.max(si)),
                     'ti_2023_max': float(np.nanmax(ti)),
                     'si_2023_full_mean': float(np.mean(si_full)),
                     'ti_2023_full_mean': float(np.nanmean(ti_full)),
                     'luma_outside_limited_pct': outside/(N*W*H)*100})
        for path in (ROOT / 'runs' / name).glob('*/frames.csv'):
            frames = pd.read_csv(path)
            frames['si_2023'] = si; frames['ti_2023'] = ti
            frames['si_2023_full'] = si_full; frames['ti_2023_full'] = ti_full
            frames.to_csv(path, index=False)
    extra = pd.DataFrame(rows)
    seq = pd.read_csv(ROOT / 'sequence_metrics.csv')
    seq = seq.drop(columns=[c for c in extra.columns if c != 'sequence' and c in seq.columns])
    seq.merge(extra, on='sequence', validate='many_to_one').to_csv(ROOT / 'sequence_metrics.csv', index=False)
    pd.concat([pd.read_csv(p) for p in sorted((ROOT/'runs').glob('*/*/frames.csv'))],
              ignore_index=True).to_csv(ROOT/'frame_metrics.csv',index=False)
    print(extra.to_string(index=False))


if __name__ == '__main__':
    main()
