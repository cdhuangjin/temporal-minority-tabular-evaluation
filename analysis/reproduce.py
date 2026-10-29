"""Exploratory revision analyses of saved summaries; no model fits or raw uploads.

Reproduce: python analysis/reproduce.py --study-root results --output outputs
Original result files are read only. No model training or raw dataset download occurs.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, rankdata, spearmanr

SEED = 20261006
B = 20000
B_RHO = 100000
MODELS = ['logistic', 'xgboost', 'lightgbm', 'mlp']
CONDITIONS = ['temporal', 'random']
METRICS = ['auprc', 'worst_window_auprc']
DATASETS = ['homesite-insurance', 'ecom-offers', 'homecredit-default']
INPUTS = {
    'primary': 'full_results_v01_20260915/full_records.csv',
    'final': 'final_convergence_sensitivity_v01_20260916/final_convergence_sensitivity_records.csv',
    'drift': 'full_analysis_v01_20260915/mechanism_cluster_window_cells.csv',
    'legacy_analysis': 'full_analysis_v01_20260915/full_analysis.json',
}

def cell_effects(df):
    means = df.groupby(['dataset', 'model', 'condition'])[['auprc', 'roc_auc']].mean()
    result = []
    for d in DATASETS:
        for m in MODELS:
            v = (means.loc[(d, m, 'random')] - means.loc[(d, m, 'temporal')]) * 100
            result.append(dict(dataset=d, model=m, ap_drop_pp=float(v.auprc),
                               auc_drop_pp=float(v.roc_auc), excess_pp=float(v.auprc-v.roc_auc)))
    return pd.DataFrame(result)

def sign_flip(cells):
    v = cells.excess_pp.to_numpy().reshape(3, 4)
    observed = float(np.median(v))
    legacy = np.array([np.median(v.ravel()*s) for s in itertools.product([-1, 1], repeat=12)])
    blocked = np.array([np.median(v*np.array(s)[:, None]) for s in itertools.product([-1, 1], repeat=3)])
    return dict(median_excess_pp=observed, independent_cell_assignments=4096,
                independent_cell_p=float(np.mean(abs(legacy)>=abs(observed)-1e-10)),
                dataset_block_assignments=8,
                dataset_block_p=float(np.mean(abs(blocked)>=abs(observed)-1e-10)),
                dataset_block_statistics_pp=blocked.tolist(),
                dataset_mean_excess_pp=dict(zip(DATASETS, v.mean(axis=1).tolist())),
                assumptions='Legacy: independent symmetric cell effects. Block sensitivity: independent datasets and joint sign symmetry of each four-model effect vector. Neither follows from a median-zero null alone. Three datasets limit inference; minimum attainable two-sided block p is 0.25.')

def rho_rows(x, y):
    rx, ry = rankdata(x, axis=-1), rankdata(y, axis=-1)
    rx -= rx.mean(axis=-1, keepdims=True)
    ry -= ry.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((rx*rx).sum(axis=-1)*(ry*ry).sum(axis=-1))
    return np.divide((rx*ry).sum(axis=-1), denominator,
                     out=np.full(denominator.shape, np.nan), where=denominator>0)

def bca_numeric(df):
    x = df.numeric_standardized_mean_drift.to_numpy()
    y = df.auprc_drop_mean_across_models.to_numpy()
    obs = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0, len(x), size=(B_RHO, len(x)))
    boot = rho_rows(x[ix], y[ix])
    valid = boot[np.isfinite(boot)]
    # Midrank bias correction handles the many equal rho values in a small sample.
    # Spearman implementations may differ at machine epsilon; mathematical ties
    # must not be classified as less-than merely because of float arithmetic.
    tie = np.isclose(valid, obs, rtol=0, atol=1e-12)
    z0 = float(norm.ppf(((valid<obs-1e-12).sum()+0.5*tie.sum())/len(valid)))
    jack = np.array([spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))])
    u = jack.mean()-jack
    acceleration = float(np.sum(u**3)/(6*np.sum(u**2)**1.5))
    za = norm.ppf([.025, .975])
    adjusted = norm.cdf(z0+(z0+za)/(1-acceleration*(z0+za)))
    loo = {d: float(spearmanr(df.loc[df.dataset!=d, 'numeric_standardized_mean_drift'],
                              df.loc[df.dataset!=d, 'auprc_drop_mean_across_models']).statistic) for d in DATASETS}
    return dict(rho=obs, n_pairs=len(x), draws=B_RHO, seed=SEED,
                nonfinite_draws=int((~np.isfinite(boot)).sum()),
                degenerate_policy='Exclude undefined correlation draws and report their count; stop if fewer than 95% are finite.',
                percentile_ci95=np.quantile(valid, [.025, .975]).tolist(),
                bca_ci95=np.quantile(valid, adjusted).tolist(),
                z0=z0, bias_tie_tolerance=1e-12, bias_tie_draws=int(tie.sum()),
                acceleration=acceleration, adjusted_quantiles=adjusted.tolist(),
                jackknife_rho=jack.tolist(), leave_one_dataset_out_rho=loo,
                interpretation='IID paired-window sensitivity only; windows share datasets and may overlap. BCa does not remove dataset dependence or create causal evidence.')

def ranking_bootstrap(df):
    rng = np.random.default_rng(SEED+1)
    summaries, gaps, split_rows = [], [], []
    for d in DATASETS:
        a = np.zeros((3, 5, 2, 4, 2))
        sub = df[df.dataset==d]
        for w in range(3):
            for s in range(5):
                for c, condition in enumerate(CONDITIONS):
                    for m, model in enumerate(MODELS):
                        r = sub[(sub.temporal_split==f'sliding-window-{w}') & (sub.seed==s) &
                                (sub.condition==condition) & (sub.model==model)]
                        if len(r)!=1: raise ValueError((d,w,s,condition,model,len(r)))
                        a[w,s,c,m,:] = r[METRICS].to_numpy()[0]
        # Preserve all model, condition and metric pairings in each seed draw.
        wi = rng.integers(0, 3, (B, 3))
        si = rng.integers(0, 5, (B, 3, 5))
        boot = a[wi[:,:,None], si].mean(axis=(1,2))  # B x condition x model x metric
        observed = a.mean(axis=(0,1))
        for k, metric in enumerate(METRICS):
            for c, condition in enumerate(CONDITIONS):
                order = np.argsort(-observed[c,:,k], kind='stable')
                orders = np.argsort(-boot[:,c,:,k], axis=1, kind='stable')
                summaries.append(dict(dataset=d,condition=condition,metric=metric,
                    observed_order=' > '.join(MODELS[i] for i in order),
                    observed_means={m: float(observed[c,i,k]) for i,m in enumerate(MODELS)},
                    exact_observed_order_frequency=float(np.mean(np.all(orders==order,axis=1))),
                    winner_frequencies={m:float(np.mean(orders[:,0]==i)) for i,m in enumerate(MODELS)},
                    draws=B,seed=SEED+1))
                pairs = [(int(order[0]), int(order[1])), (0,3)]
                for i,j in pairs:
                    v = boot[:,c,i,k]-boot[:,c,j,k]
                    gaps.append(dict(dataset=d, condition=condition, metric=metric,
                        model_a=MODELS[i],model_b=MODELS[j],gap_pp=float((observed[c,i,k]-observed[c,j,k])*100),
                        lower_pp=float(np.quantile(v,.025)*100),upper_pp=float(np.quantile(v,.975)*100),
                        positive_draw_frequency=float(np.mean(v>0))))
                    for w in range(3):
                        split_rows.append(dict(dataset=d,condition=condition,metric=metric,window=w,
                            model_a=MODELS[i],model_b=MODELS[j],
                            mean_seed_gap_pp=float((a[w,:,c,i,k]-a[w,:,c,j,k]).mean()*100)))
    return summaries,pd.DataFrame(gaps),pd.DataFrame(split_rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args=parser.parse_args(); root=args.study_root; out=args.output
    out.mkdir(parents=True, exist_ok=True)
    inputs={k:root/v for k,v in INPUTS.items()}
    primary=pd.read_csv(inputs['primary']); final=pd.read_csv(inputs['final'])
    drift=pd.read_csv(inputs['drift']); legacy=json.loads(inputs['legacy_analysis'].read_text())
    baseline=primary[primary.run_family=='baseline'].copy()
    if len(baseline)!=360 or len(final)!=360 or len(drift)!=9: raise ValueError('Unexpected inputs')
    for df in (baseline, final):
        if df[['dataset','model','condition','temporal_split','seed']].duplicated().any(): raise ValueError('Duplicate run keys')
        if not (df.status=='PASS').all(): raise ValueError('Unsuccessful run')
    # Existing original analysis is reproduced from full precision CSV rather than rounded paper values.
    primary_cells=cell_effects(baseline); final_cells=cell_effects(final)
    primary_cells.to_csv(out/'primary_cell_effects_reproduced.csv',index=False)
    final_cells.to_csv(out/'final_cell_effects.csv',index=False)
    sf={ 'primary': sign_flip(primary_cells), 'final':sign_flip(final_cells) }
    if not np.isclose(sf['primary']['independent_cell_p'],legacy['exact_sign_flip_excess']['p_two_sided']):
        raise ValueError('Legacy p not reproduced')
    if not np.isclose(sf['primary']['median_excess_pp'],legacy['exact_sign_flip_excess']['observed_median_pp'],atol=1e-7):
        raise ValueError('Legacy median not reproduced')
    # Drift values below remain the original primary-fit exploratory analysis.
    drift_rows=legacy['mechanism_cluster_associations']
    pd.DataFrame(drift_rows).to_csv(out/'all_four_drift_associations.csv',index=False)
    for row in drift_rows:
        r=spearmanr(drift[row['mechanism']],drift.auprc_drop_mean_across_models)
        if not np.isclose(r.statistic,row['spearman_rho']): raise ValueError('Drift mismatch')
    rho=bca_numeric(drift)
    if rho['nonfinite_draws']>B_RHO*.05: raise ValueError('Excessive undefined resamples')
    rankings,gaps,split_gaps=ranking_bootstrap(final)
    gaps.to_csv(out/'final_paired_ranking_gaps.csv',index=False)
    split_gaps.to_csv(out/'final_per_window_ranking_gaps.csv',index=False)
    prevalence=baseline[baseline.model=='logistic'][['dataset','temporal_split','condition','train_n','val_n','test_n','feature_dim','train_prevalence','positive_prevalence']].drop_duplicates()
    if len(prevalence)!=18: raise ValueError('Prevalence varies across seeds unexpectedly')
    prevalence['test_positive_n']=np.rint(prevalence.test_n*prevalence.positive_prevalence).astype(int)
    prevalence.to_csv(out/'split_sizes_prevalence.csv',index=False)
    window_audit={}
    for name,df in [('primary',baseline),('final',final)]:
        arr=np.array([json.loads(s) for s in df.window_auprc])
        window_audit[name]=dict(fits=len(df), bins_per_fit=5, undefined_bins=int((~np.isfinite(arr)).sum()),
            supplied_minimum_matches_all_fits=bool(np.allclose(np.nanmin(arr,axis=1),df.worst_window_auprc)))
        if arr.shape!=(360,5) or not window_audit[name]['supplied_minimum_matches_all_fits']: raise ValueError('Window mismatch')
    manifest={k:dict(relative_source=INPUTS[k],sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in inputs.items()}
    result=dict(analysis_type='Exploratory revision sensitivity, computed from saved original summaries',
        seed=SEED,environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        inputs=manifest,sign_flip=sf,numeric_rho_sensitivity=rho,rankings=rankings,window_audit=window_audit,
        ranking_interpretation='Conditional descriptive resampling: sample three matched windows with replacement, then five matched seeds in each sampled window; all four models, two conditions and two metrics share the draw. Overlap between windows and fixed benchmark selection are not removed. Quantile ranges and winner frequencies are not deployment probabilities or validated population confidence intervals.',
        limits=['No model retraining or new test-set decisions.', 'Per-row predictions unavailable, so common-prevalence reweighted AP cannot be reconstructed.', 'Matched row universe does not imply matched test prevalence, calendar periods or entity disjointness.', 'Original implementation fills numeric NaNs with raw zero before train-derived standardization; historical mean-imputation wording is incorrect.'])
    (out/'revision_sensitivity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Revision statistical evidence — exploratory supplemental analyses','',
        'Computed locally from preserved study summaries. Original data, models and results were not overwritten. These analyses were added during revision; they were not prespecified original tests.','',
        '## Dependence-aware sign-flip sensitivity','',
        '| Layer | Median excess drop (pp) | Legacy independent-cell p, 4096 assignments | Dataset-block p, 8 assignments |',
        '|---|---:|---:|---:|']
    for layer,r in sf.items(): lines.append(f"| {layer} | {r['median_excess_pp']:.4f} | {r['independent_cell_p']:.6f} | {r['dataset_block_p']:.3f} |")
    lines += ['',sf['final']['assumptions'],'Neither nonsignificant result establishes absence or equivalence.','',
        '## Final paired ranking gaps','',
        'Ranges below are 2.5%–97.5% quantiles of conditional matched window/seed resampling, not population confidence bounds. Three overlapping windows and five computational seeds do not give 15 independent cohorts.','',
        '| Dataset | Regime | Metric | Contrast | Observed gap (pp) | Resampling range (pp) |',
        '|---|---|---|---|---:|---:|']
    for r in gaps.to_dict('records'):
        lines.append(f"| {r['dataset']} | {r['condition']} | {r['metric']} | {r['model_a']} − {r['model_b']} | {r['gap_pp']:.4f} | [{r['lower_pp']:.4f}, {r['upper_pp']:.4f}] |")
    lines += ['', 'Complete observed mean-AP ordering differs between regimes for E-Commerce only. A changed full ordering must be distinguished from a change in the leading model. Small tree-model margins do not establish practically secure superiority.','',
        '## Drift association sensitivity','', '| Drift family | rho | Original raw p | Original BH q (four families) |', '|---|---:|---:|---:|']
    for r in drift_rows: lines.append(f"| {r['mechanism']} | {r['spearman_rho']:.4f} | {r['p_raw']:.4f} | {r['q_bh']:.4f} |")
    lines += ['',f"Numeric rho, nine dataset–window pairs: {rho['rho']:.4f}; new paired percentile range {rho['percentile_ci95']}; BCa range {rho['bca_ci95']}. Seed {SEED}, {B_RHO:,} draws, {rho['nonfinite_draws']} undefined draws excluded. Midrank bias correction, leave-one-pair-out acceleration={rho['acceleration']:.6f}.",
        'Original percentile interval was [0.1579, 1.0000], based on the original separate bootstrap settings. The new percentile range is disclosed separately, not substituted into history.',
        f"Leave-one-dataset-out descriptive rho: {rho['leave_one_dataset_out_rho']}. BCa assumes exchangeable pairs and does not resolve dataset dependence. The association is exploratory and not a causal mechanism.",'',
        '## Prevalence and worst-window interpretation','',
        'See split_sizes_prevalence.csv for all 18 regime/window partitions. Test prevalence is the random-ranking AP reference level, not a common held-out population. AP/AUROC summaries cannot recover AP at a standardized prevalence without per-row scores.',
        f"All saved baseline/final five-bin minima reproduce: {window_audit}. Each regime sorts its own test rows by time, makes five nearly equal-count bins, and takes minimum finite AP, then summaries average these per-fit minima. Bins are condition-specific, not matched calendar intervals.",'',
        '## Historical preprocessing clarification','',
        'temporal_minority_pilot.py computes nonmissing numeric means/SDs on training rows, fills all partitions’ NaNs with raw zero, and then standardizes using those train statistics. The accepted manuscript discloses this executed preprocessing. Earlier protocol wording should not be interpreted as evidence of mean imputation; the saved scores are unchanged.',
        'First convergence sensitivity targets 180 LR/MLP records: 138 warning-bearing records were refitted and 42 records retained. Final sensitivity resolves the remaining 78 MLP warnings; it does not refit every original model.', '',
        '## Reproduction','',
        '`python analysis/reproduce.py --study-root results --output outputs`',
        'Exact source hashes, revision-analysis package versions, bootstrap settings and all derived tables are saved with revision_sensitivity.json. These package versions describe this new analysis, not the original training environment.']
    (out/'revision_statistical_evidence.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(sign_flip=sf,numeric_rho_sensitivity=rho,ecom_mean_gaps=gaps[(gaps.dataset=='ecom-offers')&(gaps.metric=='auprc')].to_dict('records'),window_audit=window_audit),indent=2))

if __name__=='__main__': main()
