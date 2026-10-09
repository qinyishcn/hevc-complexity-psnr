"""Independent FFmpeg cross-checks plus acceptance of all retained samples."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from scipy import ndimage
from experiment import ROOT, N, W, H, FPS, SEQUENCES, command, write_json, probe
from metrics import pq_luma


def reference_transfer_check():
    # Execute only the two numerical functions from the saved MIT reference,
    # without importing its video-I/O dependencies or running module code.
    source=ROOT/'sources'/'vqeg_siti.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='SiTiCalculator')
    functions=[]
    for node in cls.body:
        if isinstance(node,ast.FunctionDef) and node.name in ['eotf_1886','oetf_pq']:
            node.decorator_list=[];functions.append(node)
    namespace={'np':np,'DEFAULT_GAMMA':2.4}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    y=np.arange(256,dtype=np.float64)
    scaled=np.clip((y-16)/219,0,1)
    ref=255*namespace['oetf_pq'](namespace['eotf_1886'](scaled,gamma=2.4,l_min=.01,l_max=300))
    error=float(np.max(np.abs(ref-pq_luma(y))))
    assert error<1e-9,error
    full_reference=255*namespace['oetf_pq'](namespace['eotf_1886'](y/255,gamma=2.4,l_min=.01,l_max=300))
    full_error=float(np.max(np.abs(full_reference-pq_luma(y,color_range='full'))))
    assert full_error<1e-9,full_error
    return {'all_256_luma_codes_checked':True,'max_abs_transfer_error':error,
            'full_range_max_abs_transfer_error':full_error,
            'reference_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'note':'BT.1886 standalone function with Lmin=.01, Lmax=300; not the reference package simplified display wrapper.'}


def crosscheck(name,rate=300):
    folder=ROOT/'validation'/name;folder.mkdir(parents=True,exist_ok=True)
    ref=ROOT/'data'/name/'reference.y4m'
    encoded=ROOT/'runs'/name/str(rate)/'encoded.mp4'
    command(['ffmpeg','-hide_banner','-y','-i',ref,'-vf',
             'setparams=range=full,siti,metadata=print:file=siti.txt','-f','null','NUL'],folder,'ffmpeg_siti')
    text=(folder/'siti.txt').read_text(encoding='utf-8')
    si=np.asarray(list(map(float,re.findall(r'lavfi.siti.si=([\d.eE+-]+)',text))))
    ti=np.asarray(list(map(float,re.findall(r'lavfi.siti.ti=([\d.eE+-]+)',text))))
    own=pd.read_csv(ROOT/'data'/name/'complexity.csv')
    assert len(si)==len(ti)==N
    si_error=float(np.max(np.abs(si-own.si)));ti_error=float(np.max(np.abs(ti[1:]-own.ti.iloc[1:])))
    # FFmpeg metadata serializes SI/TI to two decimal places.
    assert si_error<=.0051 and ti_error<=.0051,(si_error,ti_error)
    command(['ffmpeg','-hide_banner','-y','-i',encoded,'-i',ref,'-filter_complex',
             '[0:v]settb=AVTB,setpts=PTS-STARTPTS[a];[1:v]settb=AVTB,setpts=PTS-STARTPTS[b];'
             '[a][b]psnr=stats_file=psnr.txt:stats_version=2:shortest=1',
             '-f','null','NUL'],folder,'ffmpeg_psnr')
    lines=(folder/'psnr.txt').read_text(encoding='utf-8').splitlines()[1:]
    assert len(lines)==N
    ff_y=np.array([float(re.search(r'psnr_y:([\d.]+)',line)[1]) for line in lines])
    ff_mse=np.array([float(re.search(r'mse_y:([\d.]+)',line)[1]) for line in lines])
    frames=pd.read_csv(ROOT/'runs'/name/str(rate)/'frames.csv')
    error=float(np.max(np.abs(ff_y-frames.psnr_y)))
    mse_error=float(np.max(np.abs(ff_mse-frames.mse_y)))
    assert error<=.00501 and mse_error<=.00501,(error,mse_error)
    logged=float(re.search(r'PSNR y:([\d.]+)',(folder/'ffmpeg_psnr.log').read_text(encoding='utf-8'))[1])
    measured=json.loads((ROOT/'runs'/name/str(rate)/'measured.json').read_text(encoding='utf-8'))
    assert abs(logged-measured['psnr_y'])<1e-5
    return {'sequence':name,'rate_kbps':rate,'frames_compared':N,
            'max_si_abs_error':si_error,'max_ti_abs_error':ti_error,
            'max_frame_psnr_error_db':error,'max_frame_mse_error':mse_error,
            'sequence_psnr_error_db':abs(logged-measured['psnr_y'])}


def main():
    seq=pd.read_csv(ROOT/'sequence_metrics.csv');frames=pd.read_csv(ROOT/'frame_metrics.csv')
    assert set(seq.sequence)==set(SEQUENCES) and len(seq)==60 and len(frames)==9000
    assert not seq[['sequence','target_kbps']].duplicated().any()
    assert (seq.bitrate_error_pct.abs()<=1).all()
    assert np.isfinite(seq[['si_2023_mean','ti_2023_mean','si_2023_full_mean','ti_2023_full_mean','psnr_y','psnr_yuv']]).all().all()
    assert np.isfinite(frames[['si','si_2023','si_2023_full','psnr_y','psnr_yuv']]).all().all()
    assert frames.ti.isna().sum()==60 and frames.ti_2023.isna().sum()==60 and frames.ti_2023_full.isna().sum()==60
    assert (frames[frames.ti.isna()].frame==0).all()
    for (name,rate),g in frames.groupby(['sequence','target_kbps']):
        assert len(g)==N and g.frame.tolist()==list(range(N))
        assert g[g.pict_type=='I'].frame.tolist()==[0,60,120]
        assert np.isclose(g.packet_bytes.sum()*8/(N/FPS)/1000,g.actual_kbps.iloc[0])
        s=seq[(seq.sequence==name)&(seq.target_kbps==rate)].iloc[0]
        assert np.isclose(10*np.log10(255**2/g.mse_y.mean()),s.psnr_y)
        ref=ROOT/'data'/name/'reference.y4m'
        manifest=json.loads((ref.parent/'manifest.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(ref.read_bytes()).hexdigest()==manifest['sha256_reference']
        recorded=json.loads((ROOT/'runs'/name/str(rate)/'measured.json').read_text(encoding='utf-8'))
        assert hashlib.sha256((ROOT/'runs'/name/str(rate)/'encoded.mp4').read_bytes()).hexdigest()==recorded['sha256_encoded']
    checks=[crosscheck(name) for name in ['akiyo','bus','bridge_close']]
    result={'status':'passed','sequences':20,'encodes':60,'unique_source_frames':3000,
            'decoded_frame_measurements':9000,'max_abs_bitrate_error_pct':float(seq.bitrate_error_pct.abs().max()),
            'frame_counts_formats_and_sha256':'all passed','fixed_I_frames':[0,60,120],
            'reference_transfer':reference_transfer_check(),'ffmpeg_crosschecks':checks}
    write_json(ROOT/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
