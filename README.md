# Model Selection Robustness Audit — IEEE JBHI Submission
 
> **A leakage-controlled, reproducibility-hardened pipeline for auditing clinical ML model selection stability under MCAR missingness and prevalence shift.**
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [Research Questions](#research-questions)
- [Repository Structure](#repository-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Datasets](#datasets)
- [Models & Protocols](#models--protocols)
- [Evaluation Metrics & Ranking Rule](#evaluation-metrics--ranking-rule)
- [Leakage-Control Design](#leakage-control-design)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Dependencies](#dependencies)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
---
 
## Overview
 
This repository contains the full experimental pipeline for an audit study investigating whether **model selection decisions** (i.e., which classifier is chosen as the "winner") remain **stable and reproducible** under realistic clinical data perturbations:
 
1. **MCAR missingness** (up to 70% feature masking)
2. **Training prevalence / label-prior shift** (10%–50%)
3. **External cohort transfer** (MIMIC → eICU)
The pipeline is designed to be **leakage-free**, **seed-stable**, and consistent across all phases, ensuring that every ranking comparison is methodologically valid for peer-reviewed publication.
 
> **Note:** Datasets are not included in this repository. See the [Datasets](#datasets) section for acquisition instructions.
 
---
 
## Research Questions
 
| ID | Question |
|----|----------|
| **RQ1** | Does the winning model remain consistent across CV splits (S1 vs S2) and preprocessing protocols (P0–P3)? |
| **RQ2** | How stable is the full model ranking across 20 independent random seeds? |
| **RQ3** | What is the **robustness envelope** — the maximum perturbation severity before winner flip occurs? |
| **RQ4** | Are near-tie margins (AUROC/AP/Brier differences) the primary driver of instability? |
 
---
 
## Repository Structure
 
```
.
├── Code/
│   ├── phase1_3_main_audit_runner.py              # CORE: Phase 1–3 main runner
│   ├── phase4a_missingness_severity_sweep.py      # Phase 4a: MCAR severity sweep (0–70%)
│   ├── phase4b_prevalence_shift_sweep.py          # Phase 4b: Prevalence shift sweep
│   ├── phase5_tieaware_envelope_margin_analysis.py# Phase 5: Tie-aware + envelope + margin analysis
│   ├── phase5_figure_generator.py                 # Phase 5: Paper-quality figure generator
│   ├── final_summary_report.py                    # Final: Aggregate report (MD + XLSX + JSON)
│   ├── run_all.bat                                # One-command runner (Windows CMD)
│   └── run_all.sh                                 # One-command runner (Linux / macOS / Git Bash)
│
└── External Cohort/
    └── Code/
        ├── phase1_3_main_audit_runner.py          # Same runner (copy for external cohort)
        └── external_cohort_eicu_leakguard_runner.py # eICU leakage-guarded runner
```
 
**Output directories** (created automatically at runtime):
 
```
Code/
├── results/                        # Phase 1–3 outputs
├── PHASE4_SEVERITY_SWEEP/          # Phase 4a outputs
├── PHASE4B_PREVALENCE_SHIFT/       # Phase 4b outputs
├── PHASE5_ANALYSIS/                # Phase 5 outputs
├── PHASE5_FIGURES/                 # Publication-quality figures
└── final_summary/                  # Master report (MD + XLSX + CSV + JSON)
```
 
---
 
## Pipeline Architecture
 
The pipeline is structured into **5 sequential phases** followed by a final summary:
 
```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4a / 4b ──► Phase 5 ──► Summary
  │            │            │               │               │           │
Main         Repro       Synthetic      Severity         Tie-aware   Aggregate
Matrix       (20 seeds)  Control        Sweep            Envelope    Report
(A × S × P)  (P0 only)  (Dataset B)    (MCAR / Prev)    + Margin    (all phases)
```
 
### Phase 1 — Main Matrix
- Dataset A × (S1, S2) × (P0–P3) × 5 models
- Fixed seed `2026`; 5-fold outer CV, 3-fold inner CV
- Produces: `metrics_all.csv`, `summary_by_config.csv`, `winner_by_config.csv`
### Phase 2 — Reproducibility Audit
- Dataset A × (S1, S2) × P0 × **20 seeds** × 5 models
- Measures winner flip % across seeds → answers RQ2/RQ3
- Produces: `winner_flip_summary.csv`, `winner_by_seed_phase2.csv`
### Phase 3 — Synthetic Control
- Dataset B (clean + MCAR 15%) × S1 × (P0, P1) × 5 models
- Validates that P0 vs P1 differences are not dataset-specific artifacts
### Phase 4a — MCAR Severity Sweep
- Sweeps missingness rate ∈ {0, 5, 10, 20, 30, 40, 50, 60, 70%}
- 20 seeds × (S1, S2) per rate → Kendall τ and winner flip tracked
- Defines **robustness envelope**: max rate before Kendall τ < 0.8 or flip% > 5%
### Phase 4b — Prevalence Shift Sweep
- Sweeps training prevalence ∈ {native, 10%, 20%, 30%, 40%, 50%}
- Outer-test distribution always held unshifted
- Same envelope + flip onset logic as Phase 4a
### Phase 5 — Tie-Aware Analysis (Postprocessing Only)
- Reprocesses Phase 4a/4b summaries with **epsilon-tie thresholds**:
  - `ε_AUROC = 1e-3`, `ε_AP = 1e-3`, `ε_Brier = 1e-4`
- Sweeps envelope sensitivity: `flip_thr ∈ {0, 5, 10}`, `τ_thr ∈ {0.7, 0.8, 0.9}`
- Computes margin distributions (rank1 − rank2) for AUROC, AP, Brier
### Final Summary
- Aggregates all phases into a single report
- Outputs: Markdown narrative, multi-sheet Excel (with conditional formatting), CSV, JSON
---
 
## Datasets
 
> Datasets are **not included** in this repository and must be obtained separately.
 
| Tag | Description | Label | Notes |
|-----|-------------|-------|-------|
| **A** | MIMIC-derived in-hospital mortality cohort | `label_mortality` | Place as `full_analytic_dataset_mortality_all_admissions.csv` in `Code/` |
| **B** | Synthetic lipid panel dataset (1,500 patients) | TG4h ≥ global P75 | Place as `Synthetic_Dataset_1500_Patients_precise.csv` in `Code/` |
| **eICU** | User-provided eICU cohort CSV | `label` / `label24h` | Pass via `--data_path` argument to external runner |
 
Dataset A drops `hadm_id` and `discharge_location` before training. eICU additionally applies a full leakage guard (see [Leakage-Control Design](#leakage-control-design)).
 
---
 
## Models & Protocols
 
### Models (5, consistent across all phases)
 
| Key | Algorithm | Notes |
|-----|-----------|-------|
| `lr_l2` | Logistic Regression (L2) | `max_iter=5000` |
| `svm_linear_cal` | Linear SVC + Platt calibration | via `PrefitCalibrator` |
| `rf` | Random Forest | 600 estimators |
| `xgb` | XGBoost | gradient-boosted trees |
| `extratrees` | Extra-Trees | 600 estimators |
 
### Preprocessing Protocols
 
| Protocol | Description | Leakage Level |
|----------|-------------|---------------|
| **P0** | Fold-safe imputation + scaling (baseline) | None (clean) |
| **P1** | Global imputation, fold-safe scaling | Low — imputation leaks global stats |
| **P2** | Global scaling only (NaN preserved for fold imputer) | Low — scaling leaks |
| **P3** | Global transform + mutual-information feature selection | High — full global + MI selection leaks |
 
### CV Splits
 
| Key | Splitter | Use Case |
|-----|----------|----------|
| **S1** | `StratifiedKFold(5)` | Standard stratified split |
| **S2** | `GroupKFold(5)` | Subject-level group isolation (prevents patient leakage) |
 
---
 
## Evaluation Metrics & Ranking Rule
 
All models are evaluated per outer fold using three metrics:
 
| Metric | Direction | Formula |
|--------|-----------|---------|
| **AUROC** | ↑ Higher is better | `roc_auc_score(y_te, p)` |
| **AP** | ↑ Higher is better | `average_precision_score(y_te, p)` |
| **Brier Score** | ↓ Lower is better | `brier_score_loss(y_te, clip(p, 0, 1))` |
 
**Winner ranking rule (lexicographic, consistent across all phases):**
 
```
rank_key = (AUROC ↑, AP ↑, −Brier ↑)
```
 
Ties within ε-thresholds are reported as co-winner sets in Phase 5.
 
---
 
## Leakage-Control Design
 
### Structural (all phases)
- All imputation, scaling, and encoding are fitted **inside outer-train folds only** (P0 style)
- Calibration split is performed **before** hyperparameter tuning (`calibration_split_indices`)
- Nested tuning uses **only `train_sub`** — calibration set (`cal_sub`) is never seen during grid search
### MCAR Injection (Phase 4a)
- Train and test masking use **separate RNG seeds** to prevent statistical coupling
- Missingness applied to **numeric columns only**
### Prevalence Shift (Phase 4b)
- Shift applied only to **outer-train fold** — test distribution is always the native prevalence
- `subsample_to_prevalence()` downsamples only the majority class, enforcing `min_class_count=50`
### eICU Leakage Guard (6 rules)
```
(A) Drop post-outcome / discharge-derived columns (name-based regex)
(B) Drop ID-like columns (patientunitstayid, stay_id, etc.)
(C) Drop high-cardinality near-unique columns (unique_ratio ≥ 0.98)
(D) Drop sibling label columns (label24h, label48h, etc.)
(E) Optional: keep_regex whitelist for early-available features only
(F) Save leakage_guard_report.csv for full audit trail
```
 
The guard is **deterministic and label-free** — no `y` is used to decide feature drops.
 
---
 
## How to Run
 
### Prerequisites
 
```bash
pip install numpy pandas scikit-learn xgboost scipy tqdm matplotlib openpyxl
```
 
> **Python ≥ 3.9** and **scikit-learn ≥ 1.2** required. XGBoost ≥ 1.7 recommended.
 
Place both dataset CSV files inside the `Code/` directory before running.
 
### One-command full pipeline
 
**Windows CMD:**
```cmd
cd Code
run_all.bat
```
 
**Linux / macOS / Git Bash:**
```bash
cd Code
bash run_all.sh
```
 
Both scripts run all 6 steps automatically and produce a final summary report.
 
**Options:**
 
| Flag | Description |
|------|-------------|
| `--skip-phase4` | Skip Phase 4a/4b/5 (run Phase 1-3 + summary only) |
| `--skip-figures` | Skip figure generation |
| `--with-external` | Also run eICU external cohort runner |
| `--resume-from N` | Resume from step N if a previous run crashed |
 
Example:
```cmd
run_all.bat --resume-from 3
run_all.bat --skip-phase4
```
 
### Step-by-step (manual)
 
#### Phase 1–3: Main Audit
```bash
python phase1_3_main_audit_runner.py
```
Outputs saved to `results/`.
 
#### Phase 4a: MCAR Severity Sweep
```bash
python phase4a_missingness_severity_sweep.py
```
Outputs saved to `PHASE4_SEVERITY_SWEEP/`.
 
#### Phase 4b: Prevalence Shift Sweep
```bash
python phase4b_prevalence_shift_sweep.py
```
Outputs saved to `PHASE4B_PREVALENCE_SHIFT/`.
 
#### Phase 5: Tie-Aware + Envelope Analysis
```bash
python phase5_tieaware_envelope_margin_analysis.py
```
Outputs saved to `PHASE5_ANALYSIS/`.
 
#### Figure Generation
```bash
python phase5_figure_generator.py \
    --phase5_dir PHASE5_ANALYSIS \
    --out_dir PHASE5_FIGURES
```
All figures saved to `PHASE5_FIGURES/` at 300 DPI.
 
#### Final Summary Report
```bash
python final_summary_report.py
```
Outputs saved to `final_summary/`.
 
#### External Cohort (eICU)
```bash
# Pre-flight leakage check only (no training)
python "External Cohort/Code/external_cohort_eicu_leakguard_runner.py" \
    --data_path /path/to/eicu_cohort.csv \
    --label_col label \
    --group_col hospitalid \
    --preflight
 
# Full run
python "External Cohort/Code/external_cohort_eicu_leakguard_runner.py" \
    --data_path /path/to/eicu_cohort.csv \
    --label_col label \
    --group_col hospitalid \
    --results_dir results_external
```
 
---
 
## Outputs
 
### Phase 1–3 (`results/`)
 
| File | Description |
|------|-------------|
| `metrics_all.csv` | Per-fold metrics for all configs |
| `summary_by_config.csv` | Aggregated mean/std per (phase, dataset, split, protocol, model, seed) |
| `winner_by_config.csv` | Winning model per config |
| `winner_flip_summary.csv` | Winner flip % across seeds (RQ3) |
| `winner_by_seed_phase2.csv` | Per-seed winner listing for Phase 2 |
| `oof_P0_{S1,S2}_{model}.npz` | Out-of-fold predictions for P0 (audit trail) |
| `leakage_artifacts/*.json` | Leakage artifact logs per config |
| `config_log.jsonl` | Full config log |
 
### Phase 4a (`PHASE4_SEVERITY_SWEEP/`)
 
| File | Description |
|------|-------------|
| `severity_fold_metrics.csv` | Per-fold metrics per (split, seed, miss_rate, model) |
| `severity_summary_by_model.csv` | Aggregated metrics |
| `severity_winner_by_seed.csv` | Winner per (split, seed, miss_rate) |
| `robustness_envelope.csv` | Kendall τ, flip%, stability flag per miss_rate |
| `flip_onset.csv` | First miss_rate where winner flip is detected per split |
 
### Phase 4b (`PHASE4B_PREVALENCE_SHIFT/`)
 
| File | Description |
|------|-------------|
| `prevalence_shift_fold_metrics.csv` | Per-fold metrics |
| `prevalence_shift_summary_by_model.csv` | Aggregated metrics |
| `prevalence_shift_winner_by_seed.csv` | Winner per (split, seed, target_prev) |
| `robustness_envelope_prevalence_shift.csv` | Envelope per prevalence level |
| `flip_onset_prevalence_shift.csv` | First target_prev where flip is detected |
 
### Phase 5 (`PHASE5_ANALYSIS/`)
 
| File | Description |
|------|-------------|
| `phase5_winners_tieaware_missingness.csv` | Deterministic + tie-aware co-winners (missingness) |
| `phase5_winners_tieaware_prevalence_shift.csv` | Same for prevalence shift |
| `phase5_envelope_sensitivity_missingness.csv` | Envelope under all (flip_thr, tau_thr) combinations |
| `phase5_envelope_sensitivity_prevalence_shift.csv` | Same for prevalence shift |
| `phase5_margins_missingness.csv` | AUROC / AP / Brier margins (rank1 vs rank2) |
| `phase5_margins_prevalence_shift.csv` | Same for prevalence shift |
 
### Figures (`PHASE5_FIGURES/`)
 
| Figure | Description |
|--------|-------------|
| `Fig1_flip_curve_missingness.png` | Winner flip % (det vs tie-aware) vs miss rate |
| `Fig2_ranking_stability_missingness.png` | Kendall τ and Spearman ρ vs miss rate |
| `Fig3_margin_auc_distribution_missingness.png` | Histogram of AUROC margins |
| `Fig4–6_margin_*_vs_missingness.png` | Mean ± std margin vs miss rate (AUROC, AP, Brier) |
| `Fig7_envelope_sensitivity_heatmap_*.png` | Stability heatmap across (flip_thr, tau_thr) grid |
| `Fig8–14` | Analogous plots for prevalence shift |
| `Fig15–16_winner_identity_rates_*.png` | Stacked fraction of winner identity across seeds |
 
### Final Summary (`final_summary/`)
 
| File | Description |
|------|-------------|
| `final_summary_report.md` | Narrative report with all tables (Overleaf / GitHub friendly) |
| `final_summary_tables.xlsx` | Multi-sheet Excel with conditional formatting |
| `final_summary_stats.csv` | Flat key-value stats for quick lookup |
| `final_summary_stats.json` | Machine-readable summary dict |
| `section_*.csv` | Per-section detail tables |
 
---
 
## Dependencies
 
```
Python        >= 3.9
numpy         >= 1.24
pandas        >= 1.5
scikit-learn  >= 1.2
xgboost       >= 1.7
scipy         >= 1.9
tqdm          >= 4.60
matplotlib    >= 3.6
openpyxl      >= 3.0
```
 
Install all at once:
```bash
pip install numpy pandas scikit-learn xgboost scipy tqdm matplotlib openpyxl
```
 
---
 
## Reproducibility
 
All experiments use **fixed seeds** to ensure full reproducibility:
 
| Component | Seed |
|-----------|------|
| Phase 1 main | `2026` |
| Phase 3 synthetic | `2040` |
| Phase 2 / Phase 4 repro sweep | `1001–1020` (20 seeds) |
| eICU Phase 1 external | `3036` |
| Synthetic MCAR injection (Dataset B) | `777` |
 
- Outer CV: `OUTER_FOLDS = 5`, Inner CV: `INNER_FOLDS = 3`
- All rankings use **stable (mergesort)** sort for full determinism
- Full config log is appended to `results/config_log.jsonl` at runtime
---
