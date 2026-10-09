"""Download fixed Xiph samples, calibrate 2-pass HEVC, decode, measure.

Run from anywhere: python experiment.py [--sequences akiyo bus] [--rates 150 300 600]
Completed samples resume from measured.json; all external commands are logged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import time
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

from metrics import spatial_information, temporal_information, psnr_from_mse

ROOT = Path(__file__).resolve().parent
N, W, H, FPS = 150, 352, 288, 30
SEQUENCES = ('akiyo bridge_close bridge_far bus coastguard container flower football '
             'foreman hall_monitor highway husky ice mobile mother_daughter news '
             'silent soccer tempete waterfall').split()
BASE = 'https://media.xiph.org/video/derf/'
FRAME_BYTES = W * H * 3 // 2
RANGE_POOL = ThreadPoolExecutor(max_workers=32)


def download_range(url, start, end, path):
    if path.exists() and path.stat().st_size == end - start + 1:
        return path.read_bytes()
    for attempt in range(4):
        try:
            r = requests.get(url, headers={'Range': f'bytes={start}-{end}',
                                          'Accept-Encoding': 'identity'}, timeout=(20, 45))
            r.raise_for_status()
            assert r.status_code == 206, (r.status_code, url)
            assert r.headers['Content-Range'].startswith(f'bytes {start}-'), r.headers
            assert len(r.content) == end - start + 1, (start, end, len(r.content))
            path.write_bytes(r.content)
            return r.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def command(args, cwd, log_name):
    args = list(map(str, args))
    (cwd / (log_name + '.command.json')).write_text(json.dumps(args, indent=2), encoding='utf-8')
    start = time.monotonic()
    with (cwd / (log_name + '.log')).open('w', encoding='utf-8') as log:
        p = subprocess.run(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'{log_name} failed ({p.returncode}): ' +
                           (cwd / (log_name + '.log')).read_text(encoding='utf-8')[-2500:])
    return time.monotonic() - start


def acquire(name, links):
    folder = ROOT / 'data' / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        return json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    url = links[name + '_cif.y4m']
    # Persist bounded HTTP ranges: the source is slow on a single connection.
    parts = folder / 'ranges'; parts.mkdir(exist_ok=True)
    first = download_range(url, 0, 8191, parts / '000000000.bin')
    header = first.split(b'\n', 1)[0] + b'\n'
    total = len(header) + N * (FRAME_BYTES + 6)
    futures = [RANGE_POOL.submit(download_range, url, start, min(start + 262143, total-1),
                                parts / f'{start:09d}.bin')
               for start in range(8192, total, 262144)]
    payload_download = first + b''.join(f.result() for f in futures)
    print(f'TRANSFERRED {name}', flush=True)
    with io.BytesIO(payload_download) as raw:
        header = raw.readline()
        if not header.startswith(b'YUV4MPEG2 ') or b'W352 ' not in header or b'H288 ' not in header:
            raise ValueError(f'Unexpected header {name}: {header!r}')
        chroma = re.search(rb' C(\S+)', header)
        if chroma and chroma.group(1) not in (b'420', b'420jpeg', b'420mpeg2', b'420paldv'):
            raise ValueError(f'Expected 8-bit 420: {header!r}')
        payload = bytearray(header)
        yuv = np.empty((N, FRAME_BYTES), dtype=np.uint8)
        for i in range(N):
            marker = raw.readline()
            if not marker.startswith(b'FRAME'):
                raise ValueError(f'Missing frame {i} in {name}: {marker!r}')
            frame = raw.read(FRAME_BYTES)
            if len(frame) != FRAME_BYTES:
                raise ValueError(f'Truncated frame {i} in {name}')
            payload.extend(marker); payload.extend(frame)
            yuv[i] = np.frombuffer(frame, dtype=np.uint8)
        http = {'status': 206, 'range_start': 0, 'range_end': total-1,
                'strategy': '32 bounded parallel ranges, independently length checked'}
    source = folder / 'source_prefix.y4m'
    source.write_bytes(payload)
    # Change timing only; the sample planes are copied without conversion.
    norm_header = re.sub(rb' F\S+', b' F30:1', header)
    with (folder / 'reference.y4m').open('wb') as out:
        out.write(norm_header)
        for frame in yuv:
            out.write(b'FRAME\n'); out.write(frame.tobytes())
    luma = yuv[:, :W * H].reshape(N, H, W)
    records = []
    for i, y in enumerate(luma):
        records.append({'sequence': name, 'frame': i, 'si': spatial_information(y),
                        'ti': temporal_information(y, luma[i - 1]) if i else np.nan})
    pd.DataFrame(records).to_csv(folder / 'complexity.csv', index=False)
    manifest = {'sequence': name, 'url': url, 'source_header': header.decode().strip(),
                'frames_retained': N, 'source_prefix_bytes': len(payload),
                'sha256_source_prefix': hashlib.sha256(payload).hexdigest(),
                'sha256_reference': hashlib.sha256((folder / 'reference.y4m').read_bytes()).hexdigest(),
                'http': http, 'downloaded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'reference_fps': FPS, 'width': W, 'height': H, 'pix_fmt': 'yuv420p',
                'license_note': 'Xiph publicly available test media; no uniform open-source license. See original copyright links at https://media.xiph.org/video/derf/ .'}
    write_json(folder / 'manifest.json', manifest)
    print(f'DOWNLOADED {name} ({len(payload)/1e6:.1f} MB)', flush=True)
    return manifest


def probe(path, folder, tag, frames=False):
    args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_streams', '-show_packets', '-show_entries',
            'stream=codec_name,width,height,pix_fmt,nb_frames,avg_frame_rate,duration:packet=size,pts_time',
            '-of', 'json', str(path)]
    if frames:
        args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                '-show_entries', 'frame=pict_type,key_frame,pts_time,pkt_size', '-of', 'json', str(path)]
    (folder / (tag + '.command.json')).write_text(json.dumps(args, indent=2), encoding='utf-8')
    p = subprocess.run(args, capture_output=True, check=True, text=True)
    (folder / (tag + '.json')).write_text(p.stdout, encoding='utf-8')
    return json.loads(p.stdout)


def encode_measure(name, target):
    folder = ROOT / 'runs' / name / str(target)
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'measured.json'
    if result_path.exists():
        return json.loads(result_path.read_text(encoding='utf-8'))
    ref = ROOT / 'data' / name / 'reference.y4m'
    encoded = folder / 'encoded.mp4'
    setting = float(target)
    attempts = []
    for attempt in range(10):
        common = ['ffmpeg', '-hide_banner', '-y', '-i', ref, '-map', '0:v:0',
                  '-frames:v', N, '-an', '-c:v', 'libx265', '-preset', 'medium',
                  '-tune', 'psnr', '-pix_fmt', 'yuv420p', '-b:v', str(round(setting * 1000)),
                  '-color_range', 'tv']
        params = ('stats=rate.stats:slow-firstpass=1:keyint=60:min-keyint=60:scenecut=0:'
                  'open-gop=0:bframes=4:b-adapt=0:frame-threads=1:pools=2:info=0')
        elapsed = 0
        elapsed += command(common + ['-x265-params', 'pass=1:' + params, '-f', 'null', 'NUL'],
                           folder, f'attempt{attempt}_pass1')
        elapsed += command(common + ['-x265-params', 'pass=2:' + params, encoded],
                           folder, f'attempt{attempt}_pass2')
        info = probe(encoded, folder, f'attempt{attempt}_probe')
        stream = info['streams'][0]
        count = len(info['packets'])
        assert count == N and int(stream['nb_frames']) == N, (name, count, stream)
        assert stream['codec_name'] == 'hevc' and stream['pix_fmt'] == 'yuv420p'
        assert stream['width'] == W and stream['height'] == H and stream['avg_frame_rate'] == '30/1'
        actual = sum(int(p['size']) for p in info['packets']) * 8 / (N / FPS) / 1000
        error = actual / target - 1
        attempts.append({'attempt': attempt, 'encoder_setting_kbps': setting,
                         'actual_packet_kbps': actual, 'relative_error': error,
                         'encode_wall_seconds': elapsed})
        if abs(error) <= .01:
            break
        setting *= target / actual
        print(f'CALIBRATE {name} {target}: actual={actual:.3f}, next={setting:.3f}', flush=True)
    else:
        write_json(folder / 'failed_calibration.json', attempts)
        raise RuntimeError(f'Cannot match bitrate {name} {target}: {attempts}')
    decoded = folder / 'decoded.yuv'
    command(['ffmpeg', '-hide_banner', '-y', '-i', encoded, '-map', '0:v:0', '-an',
             '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', '-f', 'rawvideo', decoded],
            folder, 'decode')
    assert decoded.stat().st_size == N * FRAME_BYTES
    rec = np.memmap(decoded, dtype=np.uint8, mode='r', shape=(N, FRAME_BYTES))
    # Source file has no per-frame extensions after normalization.
    source_yuv = np.empty((N, FRAME_BYTES), dtype=np.uint8)
    with ref.open('rb') as inp:
        inp.readline()
        for i in range(N):
            assert inp.readline() == b'FRAME\n'
            source_yuv[i] = np.frombuffer(inp.read(FRAME_BYTES), dtype=np.uint8)
    diff = source_yuv.astype(np.float64) - rec.astype(np.float64)
    mse_y = np.mean(diff[:, :W*H] ** 2, axis=1)
    mse_yuv = np.mean(diff ** 2, axis=1)
    complexity = pd.read_csv(ROOT / 'data' / name / 'complexity.csv')
    frameinfo = probe(encoded, folder, 'decoded_frames', frames=True)['frames']
    assert len(frameinfo) == N
    assert np.isfinite(mse_y).all() and (mse_y > 0).all()
    complexity['target_kbps'] = target
    complexity['actual_kbps'] = actual
    complexity['pict_type'] = [f['pict_type'] for f in frameinfo]
    complexity['packet_bytes'] = [int(f['pkt_size']) for f in frameinfo]
    complexity['mse_y'] = mse_y
    complexity['mse_yuv'] = mse_yuv
    complexity['psnr_y'] = [psnr_from_mse(x) for x in mse_y]
    complexity['psnr_yuv'] = [psnr_from_mse(x) for x in mse_yuv]
    complexity.to_csv(folder / 'frames.csv', index=False)
    result = {'sequence': name, 'target_kbps': target, 'actual_kbps': actual,
              'bitrate_error_pct': error * 100, 'frames': N, 'duration_s': N / FPS,
              'si_max': float(complexity.si.max()), 'si_mean': float(complexity.si.mean()),
              'ti_max': float(complexity.ti.max()), 'ti_mean': float(complexity.ti.mean()),
              'psnr_y': psnr_from_mse(float(mse_y.mean())),
              'psnr_yuv': psnr_from_mse(float(mse_yuv.mean())),
              'psnr_y_frame_db_mean': float(complexity.psnr_y.mean()),
              'mse_y': float(mse_y.mean()), 'mse_yuv': float(mse_yuv.mean()),
              'calibration_attempts': attempts, 'sha256_encoded': hashlib.sha256(encoded.read_bytes()).hexdigest()}
    write_json(result_path, result)
    del rec
    decoded.unlink()  # Regenerable intermediate only; retained MP4 is the actual experiment artifact.
    print(f'DONE {name} {target}: {actual:.2f} kb/s, Y-PSNR={result["psnr_y"]:.3f} dB', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences', nargs='+', default=SEQUENCES, choices=SEQUENCES)
    parser.add_argument('--rates', nargs='+', type=int, default=[150, 300, 600])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    sources = ROOT / 'sources'; sources.mkdir(exist_ok=True)
    response = requests.get(BASE, timeout=60); response.raise_for_status()
    (sources / 'xiph_index.html').write_text(response.text, encoding='utf-8')
    links = {Path(x).name: urljoin(BASE, x)
             for x in re.findall(r'href=[\"\']([^\"\']+)', response.text)
             if x.endswith('.y4m')}
    write_json(sources / 'environment.json', {'python': platform.python_version(),
               'numpy': np.__version__, 'pandas': pd.__version__, 'platform': platform.platform(),
               'ffmpeg': subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout,
               'sequences': args.sequences, 'rates': args.rates, 'workers': args.workers})
    manifests, results, encodes = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as enc, ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(acquire, s, links) for s in args.sequences]
        for future in as_completed(futures):
            manifest = future.result(); manifests.append(manifest)
            encodes.extend(enc.submit(encode_measure, manifest['sequence'], b) for b in args.rates)
            write_json(ROOT / 'dataset_manifest.json', sorted(manifests, key=lambda x: x['sequence']))
        results = [f.result() for f in as_completed(encodes)]
    rows = [{k: v for k, v in r.items() if k != 'calibration_attempts'} for r in results]
    pd.DataFrame(rows).sort_values(['target_kbps', 'sequence']).to_csv(ROOT / 'sequence_metrics.csv', index=False)
    pd.concat([pd.read_csv(ROOT / 'runs' / s / str(b) / 'frames.csv')
               for s in args.sequences for b in args.rates], ignore_index=True).to_csv(ROOT / 'frame_metrics.csv', index=False)
    print(f'COMPLETE {len(results)} encodes', flush=True)


if __name__ == '__main__':
    main()
