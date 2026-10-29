# Temporal minority-sensitive tabular evaluation

Minimal statistical reproduction for **Temporal Evaluation Can Reshape Minority-Sensitive Model Rankings in Tabular Benchmarks: A Reproducible Matched-Split Study**, by Jin Huang and Qisen Gao, accepted by *Informatica*.

This repository recomputes results from saved, derived experiment records. It does **not** contain the manuscript, editorial materials, original benchmark rows, model checkpoints, or per-row predictions. It does not retrain models or download datasets.

## Quick start

Python 3.12 is recommended. Create and activate a virtual environment, then run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python analysis/reproduce.py --study-root results --output outputs
python verify.py --output outputs
```

The analysis uses CPU only. Its largest resampling steps use 20,000 paired ranking draws and 100,000 correlation draws. Generated files are written to `outputs/`; the committed inputs are read only. The verification command checks input hashes and compares six generated tables to the preserved reference results, including all paired ranking gaps and interval endpoints. It also checks the sign-flip statistics and worst-window accounting.

## Reproduced results

| Quantity | Expected result |
|---|---|
| Primary median excess AP-versus-AUROC contrast | -1.4266536670 percentage points |
| Final median excess contrast | -0.9608868404 percentage points |
| Independent-cell sign-flip p, primary / final | 0.55224609375 / 0.56298828125 |
| Dataset-block sign-flip p, both layers | 1.00 |
| Final complete mean-AP ordering changes | E-Commerce only |
| Final worst-window AP ordering changes | E-Commerce and HomeCredit |
| E-Commerce leading-tree gaps, temporal / random | 0.4311 / 0.0580 percentage points |

Both leading-tree gap resampling ranges include zero. The generated `revision_sensitivity.json` records full-precision statistics, runtime package versions, input hashes, and assumptions. `revision_statistical_evidence.md` is a generated results summary, not the paper.

## Contents and provenance

- `analysis/reproduce.py`: portable revision analysis, with original numerical functions retained.
- `results/full_results_v01_20260915/full_records.csv`: 450 primary fit records (360 baselines and 90 weighting interventions).
- `results/final_convergence_sensitivity_v01_20260916/final_convergence_sensitivity_records.csv`: 360 merged final baseline records.
- `results/full_analysis_v01_20260915/mechanism_cluster_window_cells.csv`: nine dataset-window drift summaries.
- `results/full_analysis_v01_20260915/full_analysis.json`: preserved primary analysis for numerical cross-checks and original drift-association values.
- `expected.json`: six reference tables retained from the revision analysis.
- `provenance.json`: result-file hashes and documented adaptation of the analysis entry point.
- `verify.py`: deterministic checks against the frozen reference tables.

The four result inputs are byte-identical to the corresponding submission-safe archived files. The public entry point adds a configurable output folder, omits two provenance-only hash inputs that were not used in calculations, and updates the historical preprocessing note and command text. No statistical formula, random seed, resampling count, or fit metric was changed.

The original benchmark is [TabReD](https://github.com/yandex-research/tabred), by Rubachev et al. ([paper](https://arxiv.org/abs/2406.19380)). The three tasks are Homesite Insurance, E-Commerce Offers, and HomeCredit Default. Original competition data and split arrays must be obtained under their source terms; they are not needed for the commands above. No original participant/customer records are included here.

## Interpretation and limits

- AUPRC means stepwise average precision (AP). Contrasts are matched-random minus temporal scores in percentage points.
- Three datasets, three matched split pairs, and five seeds are included. Windows may overlap and are not independent deployment cohorts.
- Ranking quantile ranges and resampling frequencies are conditional stability summaries, not population confidence guarantees or deployment winner probabilities.
- Worst-window AP is the minimum over five chronological bins of each fit's own test rows; conditions do not share identical test rows or calendar bins.
- The original preprocessing filled numerical NaNs with raw zero before training-statistics standardization; this is disclosed in the accepted manuscript.
- This minimal release reproduces the revision statistical analysis. It does not independently reproduce model fitting, the original 88-cell TabReD report audit, the pilot, all figures, or the original intervention bootstrap calculation. Intervention records are preserved in the primary CSV.
- Row-level prediction probabilities were not retained, so common-prevalence reweighted AP cannot be reconstructed. The pinned packages describe this analysis environment, not a complete historical model-training environment.

## Citation and license

Please cite the study title and authors above and identify the repository commit used. The journal article DOI and publication details can be added when available; none are invented here.

Author-created code and accompanying author-created materials are distributed under the [MIT license](LICENSE). This license does not grant rights to upstream TabReD/Kaggle datasets or third-party software, which retain their own terms.
