"""Sequence-level inference, clustered descriptive frame fits, publication plots."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
SEED, BOOT, PERM = 20261008, 5000, 19999
RATES = [150, 300, 600]
METRICS = ['si_2023_mean', 'ti_2023_mean', 'si_mean', 'ti_mean', 'si_max', 'ti_max',
           'si_2023_full_mean', 'ti_2023_full_mean']
LABELS = {'si_2023_mean': 'SI (P.910 2023, mean)', 'ti_2023_mean': 'TI (P.910 2023, mean)',
          'si_mean': 'SI (classical, mean)', 'ti_mean': 'TI (classical, mean)',
          'si_max': 'SI (classical, max)', 'ti_max': 'TI (classical, max)'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130, 'savefig.dpi': 180})


def row_corr(x, y):
    x = x - x.mean(axis=-1, keepdims=True); y = y - y.mean(axis=-1, keepdims=True)
    return np.sum(x*y, axis=-1) / np.sqrt(np.sum(x*x, axis=-1)*np.sum(y*y, axis=-1))


def holm(p):
    p = np.asarray(p); order = np.argsort(p); adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def correlations(x, y, rng):
    x, y = np.asarray(x), np.asarray(y); n = len(x)
    r = stats.pearsonr(x, y); rho = stats.spearmanr(x, y)
    indices = rng.integers(0, n, size=(BOOT, n))
    rb = row_corr(x[indices], y[indices])
    rhob = row_corr(stats.rankdata(x[indices], axis=1), stats.rankdata(y[indices], axis=1))
    ci = np.nanquantile(rb, [.025, .975]); ci_rho = np.nanquantile(rhob, [.025, .975])
    # The unit permuted is a whole video, never an individual frame.
    permutations = np.stack([rng.permutation(n) for _ in range(PERM)])
    null_r = row_corr(np.broadcast_to(x, (PERM, n)), y[permutations])
    null_rho = row_corr(np.broadcast_to(stats.rankdata(x), (PERM, n)),
                       stats.rankdata(y)[permutations])
    fit = stats.linregress(x, y)
    loo_r=[];loo_linear=[];loo_quadratic=[]
    for i in range(n):
        keep=np.arange(n)!=i
        loo_r.append(stats.pearsonr(x[keep],y[keep]).statistic)
        loo_linear.append(y[i]-np.polyval(np.polyfit(x[keep],y[keep],1),x[i]))
        loo_quadratic.append(y[i]-np.polyval(np.polyfit(x[keep],y[keep],2),x[i]))
    return {'n_sequences': n, 'pearson_r': float(r.statistic),
            'pearson_ci_low': float(ci[0]), 'pearson_ci_high': float(ci[1]),
            'pearson_p_two_sided': float(r.pvalue),
            'pearson_perm_p_two_sided': float((np.count_nonzero(abs(null_r) >= abs(r.statistic))+1)/(PERM+1)),
            'positive_perm_p': float((np.count_nonzero(null_r >= r.statistic)+1)/(PERM+1)),
            'spearman_rho': float(rho.statistic), 'spearman_ci_low': float(ci_rho[0]),
            'spearman_ci_high': float(ci_rho[1]),
            'spearman_perm_p_two_sided': float((np.count_nonzero(abs(null_rho) >= abs(rho.statistic))+1)/(PERM+1)),
            'linear_slope': float(fit.slope), 'linear_intercept': float(fit.intercept),
            'linear_r2': float(r.statistic ** 2),
            'leave_one_video_out_r_min':float(np.min(loo_r)),
            'leave_one_video_out_r_max':float(np.max(loo_r)),
            'linear_loo_rmse_db':float(np.sqrt(np.mean(np.square(loo_linear)))),
            'quadratic_loo_rmse_db':float(np.sqrt(np.mean(np.square(loo_quadratic))))}


def cluster_linear_band(data, xcol, grid, rng):
    d = data[[xcol, 'psnr_y', 'sequence']].dropna()
    values = []
    for _, g in d.groupby('sequence'):
        x, y = g[xcol].to_numpy(), g.psnr_y.to_numpy()
        values.append([len(x), x.sum(), y.sum(), (x*x).sum(), (x*y).sum()])
    values = np.asarray(values)
    samples = values[rng.integers(0, len(values), size=(BOOT, len(values)))].sum(axis=1)
    n, sx, sy, sxx, sxy = samples.T
    slope = (sxy - sx*sy/n)/(sxx - sx*sx/n)
    intercept = sy/n - slope*sx/n
    predictions = intercept[:, None] + slope[:, None]*grid
    return np.quantile(predictions, [.025, .975], axis=0)


def fit_curves(ax, data, xcol, rng, band=True):
    d = data[[xcol, 'psnr_y', 'sequence']].dropna()
    x, y = d[xcol].to_numpy(), d.psnr_y.to_numpy()
    grid = np.linspace(x.min(), x.max(), 180)
    linear = np.polyfit(x, y, 1); quadratic = np.polyfit(x, y, 2)
    if band:
        lo, hi = cluster_linear_band(d, xcol, grid, rng)
        ax.fill_between(grid, lo, hi, color='#284a65', alpha=.12, label='95% video-bootstrap CI')
    ax.plot(grid, np.polyval(linear, grid), color='#112d42', lw=2, label='Linear fit')
    ax.plot(grid, np.polyval(quadratic, grid), color='#b15c11', lw=1.7, ls='--', label='Quadratic (descriptive)')
    ax.grid(alpha=.18)
    return linear


def save(fig, name):
    fig.savefig(OUT/(name+'.png'), bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight')
    plt.close(fig)


def plot_panels(seq, frames, records, colours, rng):
    for level, data, cols in [('sequence', seq, ['si_2023_mean', 'ti_2023_mean']),
                               ('frame', frames, ['si_2023', 'ti_2023'])]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
        for j, rate in enumerate(RATES):
            d = data[data.target_kbps == rate]
            for i, col in enumerate(cols):
                ax = axes[i, j]
                for name, g in d.groupby('sequence'):
                    ax.scatter(g[col], g.psnr_y, s=35 if level=='sequence' else 5,
                               alpha=.9 if level=='sequence' else .24,
                               color=colours[name], edgecolors='none', rasterized=True)
                fit = fit_curves(ax, d, col, rng)
                r = stats.pearsonr(d[[col,'psnr_y']].dropna()[col], d[[col,'psnr_y']].dropna().psnr_y).statistic
                ax.set_title(f'{rate} kb/s | r={r:.3f}')
                ax.set_xlabel(('SI' if i==0 else 'TI')+' (P.910 2023; '+('video mean' if level=='sequence' else 'per frame')+')')
                if j==0: ax.set_ylabel('Decoded luma PSNR (dB)')
                ax.text(.98,.97, f'y={fit[0]:+.3f}x{fit[1]:+.2f}', ha='right', va='top',
                        transform=ax.transAxes, fontsize=9, bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5,-.01), frameon=False)
        title = '20 videos per bitrate; inference uses videos as independent units' if level=='sequence' else '3,000 frames per bitrate; r is descriptive; CI resamples whole videos'
        fig.suptitle('H.265 at matched measured average bitrate\n'+title, fontsize=14)
        fig.tight_layout(rect=[0,.045,1,.93])
        save(fig, f'{level}_scatter_fits_2023')
    # SI/TI content coverage. Names remain readable and linked to data CSVs.
    d = seq[seq.target_kbps == 300]
    fig,ax=plt.subplots(figsize=(11,7))
    offsets={'akiyo':(-8,24),'bridge_close':(14,-12),'mother_daughter':(-8,20),'waterfall':(9,13)}
    for row in d.itertuples():
        ax.scatter(row.si_2023_mean,row.ti_2023_mean,color=colours[row.sequence],s=60)
        offset=offsets.get(row.sequence,(4,4))
        ax.annotate(row.sequence,(row.si_2023_mean,row.ti_2023_mean),xytext=offset,
                    textcoords='offset points',fontsize=8,ha='right' if offset[0]<0 else 'left',
                    arrowprops={'arrowstyle':'-','color':'#888','lw':.5} if row.sequence in offsets else None)
    ax.set_xlabel('SI (P.910 2023, video mean)');ax.set_ylabel('TI (P.910 2023, video mean)')
    ax.set_xlim(d.si_2023_mean.min()-8,d.si_2023_mean.max()+5)
    ax.set_title('Preselected source-content coverage (20 CIF videos)');ax.grid(alpha=.2)
    save(fig,'content_coverage')
    # Correlation sensitivity to algorithm/aggregation.
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharex=True)
    colors_rate=['#155e75','#a16207','#7e22ce']
    for i,prefix in enumerate(['si','ti']):
        subset=records[records.metric.str.startswith(prefix)]
        names=[prefix+'_2023_mean',prefix+'_2023_full_mean',prefix+'_mean',prefix+'_max']
        for j,rate in enumerate(RATES):
            g=subset[subset.target_kbps==rate].set_index('metric').loc[names]
            yy=np.arange(4)+(j-1)*.20
            axes[i].errorbar(g.pearson_r,yy,xerr=[g.pearson_r-g.pearson_ci_low,
                                               g.pearson_ci_high-g.pearson_r],
                            fmt='o',capsize=3,label=f'{rate} kb/s',color=colors_rate[j])
        axes[i].axvline(0,color='#555',ls='--');axes[i].set_xlim(-1.05,1.05)
        axes[i].set_yticks(range(4),['2023 limited (primary)','2023 full range','Classical mean','Classical max'])
        axes[i].set_xlabel('Pearson r with video PSNR-Y (95% video-bootstrap CI)')
        axes[i].set_title(prefix.upper()+' sensitivity');axes[i].grid(axis='x',alpha=.2)
        axes[i].legend(frameon=False)
    fig.tight_layout();save(fig,'correlation_robustness')
    # Actual bitrate controls and PSNR range.
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for rate,c in zip(RATES,colors_rate):
        g=seq[seq.target_kbps==rate].sort_values('sequence')
        axes[0].scatter(g.sequence,g.bitrate_error_pct,label=f'{rate} kb/s',s=28,color=c)
        axes[1].scatter(g.actual_kbps,g.psnr_y,label=f'{rate} kb/s',s=35,color=c)
    axes[0].axhspan(-1,1,color='#65a30d',alpha=.10)
    axes[0].axhline(0,color='#666',lw=1)
    axes[0].set_xticks(range(len(d)),sorted(d.sequence),rotation=75,ha='right',fontsize=8)
    axes[0].set_ylabel('Measured video bitrate error (%)');axes[0].set_title('Acceptance interval: target +/-1%')
    axes[1].set_xlabel('Measured video packet bitrate (kb/s)');axes[1].set_ylabel('PSNR-Y (dB)')
    axes[1].set_title('Quality spread at matched average bitrate')
    for ax in axes:ax.legend(frameon=False);ax.grid(alpha=.18)
    fig.tight_layout();save(fig,'bitrate_quality_audit')


def joint_models(seq, frames):
    rows=[]
    for rate in RATES:
        d=seq[seq.target_kbps==rate].copy()
        # HC3 robust covariance at independent-video level.
        x=np.column_stack([np.ones(len(d)),stats.zscore(d.si_2023_mean),stats.zscore(d.ti_2023_mean),
                           d.bitrate_error_pct.to_numpy()])
        y=d.psnr_y.to_numpy(); beta=np.linalg.lstsq(x,y,rcond=None)[0]
        inv=np.linalg.pinv(x.T@x);res=y-x@beta;hat=np.sum((x@inv)*x,axis=1)
        meat=x.T@((res/(1-hat))[:,None]**2*x);cov=inv@meat@inv;se=np.sqrt(np.diag(cov))
        for i,name in enumerate(['intercept','si_z','ti_z','bitrate_error_pct']):
            rows.append({'target_kbps':rate,'model':'sequence_joint_HC3','term':name,'coef':beta[i],
                         'se':se[i],'ci_low':beta[i]-stats.t.ppf(.975,len(d)-x.shape[1])*se[i],
                         'ci_high':beta[i]+stats.t.ppf(.975,len(d)-x.shape[1])*se[i],
                         'p_two_sided':2*stats.t.sf(abs(beta[i]/se[i]),len(d)-x.shape[1])})
        # Within-video frame association, controlling picture type; cluster at video.
        f=frames[frames.target_kbps==rate].dropna(subset=['ti_2023']).copy()
        z=np.column_stack([stats.zscore(f.si_2023),stats.zscore(f.ti_2023),
                           (f.pict_type=='I').astype(float),(f.pict_type=='P').astype(float)])
        # Demean all regressors and outcome within each video (video fixed effects).
        group=f.sequence.to_numpy(); yy=f.psnr_y.to_numpy().copy()
        for name in np.unique(group):
            ix=group==name; z[ix]-=z[ix].mean(axis=0); yy[ix]-=yy[ix].mean()
        beta=np.linalg.lstsq(z,yy,rcond=None)[0];inv=np.linalg.pinv(z.T@z);res=yy-z@beta
        meat=np.zeros((z.shape[1],z.shape[1]));clusters=np.unique(group)
        for name in clusters:
            ix=group==name;u=z[ix].T@res[ix];meat+=np.outer(u,u)
        k=z.shape[1]+len(clusters);cov=inv@meat@inv*len(clusters)/(len(clusters)-1)*(len(f)-1)/(len(f)-k)
        se=np.sqrt(np.diag(cov));critical=stats.t.ppf(.975,len(clusters)-1)
        for i,name in enumerate(['si_z','ti_z','I_frame','P_frame']):
            rows.append({'target_kbps':rate,'model':'frame_within_video_clustered','term':name,'coef':beta[i],
                         'se':se[i],'ci_low':beta[i]-critical*se[i],'ci_high':beta[i]+critical*se[i],
                         'p_two_sided':2*stats.t.sf(abs(beta[i]/se[i]),len(clusters)-1)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    seq=pd.read_csv(ROOT/'sequence_metrics.csv');frames=pd.read_csv(ROOT/'frame_metrics.csv')
    assert len(seq)==60 and len(frames)==9000 and seq.sequence.nunique()==20
    assert (seq.bitrate_error_pct.abs()<=1).all()
    rng=np.random.default_rng(SEED);records=[]
    for rate in RATES:
        d=seq[seq.target_kbps==rate].sort_values('sequence')
        for metric in METRICS:
            records.append({'target_kbps':rate,'metric':metric,**correlations(d[metric],d.psnr_y,rng)})
    summary=pd.DataFrame(records)
    primary=summary.metric.isin(['si_2023_mean','ti_2023_mean'])
    summary.loc[primary,'primary_pearson_p_holm']=holm(summary.loc[primary,'pearson_perm_p_two_sided'])
    summary.loc[primary,'primary_positive_p_holm']=holm(summary.loc[primary,'positive_perm_p'])
    summary.loc[primary,'primary_spearman_p_holm']=holm(summary.loc[primary,'spearman_perm_p_two_sided'])
    summary.to_csv(ROOT/'correlations.csv',index=False)
    joint_models(seq,frames).to_csv(ROOT/'joint_regressions.csv',index=False)
    descriptive=[]
    for rate in RATES:
        f=frames[frames.target_kbps==rate]
        for selection,g in [('all_frames',f),('exclude_I_frames',f[f.pict_type!='I'])]:
            for metric in ['si_2023','ti_2023']:
                q=g[[metric,'psnr_y']].dropna()
                descriptive.append({'target_kbps':rate,'selection':selection,'metric':metric,'n_frames':len(q),
                                    'pearson_r_descriptive':stats.pearsonr(q[metric],q.psnr_y).statistic,
                                    'spearman_rho_descriptive':stats.spearmanr(q[metric],q.psnr_y).statistic})
    pd.DataFrame(descriptive).to_csv(ROOT/'frame_descriptive.csv',index=False)
    colours={name:plt.get_cmap('tab20')(i) for i,name in enumerate(sorted(seq.sequence.unique()))}
    plot_panels(seq,frames,summary,colours,rng)
    spans=[]
    for rate,g in seq.groupby('target_kbps'):
        lo=g.loc[g.psnr_y.idxmin()];hi=g.loc[g.psnr_y.idxmax()]
        spans.append({'target_kbps':int(rate),'min_sequence':lo.sequence,'min_psnr_y':float(lo.psnr_y),
                      'max_sequence':hi.sequence,'max_psnr_y':float(hi.psnr_y),
                      'spread_db':float(hi.psnr_y-lo.psnr_y),
                      'max_abs_bitrate_error_pct':float(g.bitrate_error_pct.abs().max())})
    (ROOT/'summary.json').write_text(json.dumps({'seed':SEED,'bootstrap_repetitions':BOOT,
       'permutation_repetitions':PERM,'independent_sequences':20,'unique_source_frames':3000,
       'decoded_frame_measurements':9000,'primary_method':'P.910 2023 SDR full BT.1886 Lw=300 Lb=0.01 gamma=2.4 to PQ; video means',
       'primary_correlations':summary[primary].to_dict('records'),'quality_ranges':spans},indent=2),encoding='utf-8')
    print(summary[primary][['target_kbps','metric','pearson_r','pearson_ci_low','pearson_ci_high',
                            'primary_pearson_p_holm','spearman_rho','primary_positive_p_holm']].to_string(index=False))
    print(json.dumps(spans,indent=2))


if __name__=='__main__':main()
