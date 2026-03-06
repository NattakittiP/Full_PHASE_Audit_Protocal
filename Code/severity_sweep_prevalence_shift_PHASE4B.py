# ------------------------------------------------------------
# PHASE 4b — Prevalence / Label-Shift Severity Sweep (Dataset A)
#
# Goal (Q1-hardening):
#   Stress-test model selection stability under TRAINING prevalence shift
#   (a label-prior / prevalence shift), while evaluating on the untouched
#   outer-test distribution.
#
# Design requirements (must match Phase 1–4a runner + Phase4a sweep):
#   - Leakage-controlled evaluation:
#       * preprocessing occurs inside folds (P0 style)
#       * calibration split happens BEFORE hyperparameter tuning
#       * tuning uses ONLY train_sub (not cal_sub, not test)
#       * fit on train_sub, calibrate on cal_sub
#   - Winner ranking key CONSISTENT with Phase 1–3:
#       AUROC (desc), then AP (desc), then Brier (asc)
#   - Uses the SAME helper functions imported from your runner:
#       load_dataset_A, build_preprocessor, make_model_and_grid,
#       fit_best_model_nested, calibration_split_indices, PrefitCalibrator,
#       predict_proba_safe, get_outer
#
# Outputs (mirrors Phase4a structure):
#   OUT_DIR/
#     prevalence_shift_fold_metrics.csv
#     prevalence_shift_summary_by_model.csv
#     prevalence_shift_winner_by_seed.csv
#     robustness_envelope_prevalence_shift.csv
#     flip_onset_prevalence_shift.csv
#
# Notes:
#   - This script runs on Dataset A by default (same as Phase1–4a).
#   - Shift is applied ONLY to the outer-train fold (before cal split).
#   - For S2 (GroupKFold), outer split still enforces group separation
#     between train/test. Inside outer-train, we subsample instances to
#     match target prevalence; cal split remains group-aware via the runner.
# ------------------------------------------------------------

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import kendalltau, spearmanr

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline

# ============================================================
# IMPORT SECTION (runner functions)  <<<< IMPORTANT
# ============================================================
from jcsse_audit_runner_tqdm_hardened import (
    load_dataset_A,               # returns: X, y, groups, num_cols, cat_cols
    build_preprocessor,           # fold-safe preprocessing builder
    make_model_and_grid,          # returns (model, grid, do_cal)
    fit_best_model_nested,        # nested CV tuning
    calibration_split_indices,    # train_sub / cal_sub split inside outer-train
    PrefitCalibrator,             # calibration wrapper
    predict_proba_safe,           # safe proba extraction
    get_outer,                    # outer CV splitter consistent with runner (S1/S2)
)

# ============================================================
# CONFIG
# ============================================================
OUT_DIR = "PHASE4B_PREVALENCE_SHIFT"
os.makedirs(OUT_DIR, exist_ok=True)

# Target training prevalences (label-prior shift)
# Keep a wide-but-reasonable sweep; include the "native" baseline (None)
TARGET_PREVS: List[Optional[float]] = [None, 0.10, 0.20, 0.30, 0.40, 0.50]
NATIVE_PREV_TAG = -1.0  # numeric tag used in CSVs for the native (unshifted) training prevalence


SEEDS = list(range(1001, 1021))         # 20 seeds (match Phase2/4a)
SPLITS = ["S1", "S2"]
MODELS = ["lr_l2", "svm_linear_cal", "rf", "xgb", "extratrees"]

# Stability thresholds for "robustness envelope" (same defaults as Phase4a)
MAX_WINNER_FLIP_PCT = 5.0
MIN_KENDALL_TAU = 0.8

# Minimum samples per class to avoid pathological tiny training subsets
MIN_CLASS_COUNT = 50

# ============================================================
# Winner ranking key (Phase 1–3 consistent)
# rank by AUROC desc, then AP desc, then Brier asc
# ============================================================
def rank_key(mean_auc: float, mean_ap: float, mean_brier: float) -> Tuple[float, float, float]:
    return (float(mean_auc), float(mean_ap), -float(mean_brier))

# ============================================================
# Prevalence shift utility (outer-train only)
# ============================================================
def subsample_to_prevalence(
    y: np.ndarray,
    target_prev: float,
    seed: int,
    min_class_count: int = MIN_CLASS_COUNT,
) -> np.ndarray:
    """
    Return indices (relative to y) that subsample the data to approximate
    a target prevalence (positive fraction).

    - Works by downsampling the majority class only (keeps all minority if needed).
    - Ensures at least min_class_count from each class when possible.
    - If impossible (too few examples), returns all indices (no shift).
    """
    y = np.asarray(y).astype(int).ravel()
    n = len(y)
    idx_all = np.arange(n)

    pos = idx_all[y == 1]
    neg = idx_all[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return idx_all

    rng = np.random.default_rng(seed)

    # Compute required counts if we keep all of one class and downsample the other
    # Target prev = n_pos' / (n_pos' + n_neg')
    # If we keep all positives, solve for n_neg'
    n_neg_req = int(round(n_pos * (1.0 - target_prev) / max(target_prev, 1e-12)))
    # If we keep all negatives, solve for n_pos'
    n_pos_req = int(round(n_neg * target_prev / max(1.0 - target_prev, 1e-12)))

    # Decide which class to downsample (downsample majority to meet target)
    # Case 1: target_prev <= current_prev => need relatively more negatives vs positives => downsample positives OR keep all positives?
    current_prev = n_pos / (n_pos + n_neg)

    if target_prev <= current_prev:
        # Need lower prevalence => downsample positives (if possible)
        desired_pos = min(n_pos, n_pos_req)
        desired_neg = n_neg  # keep all negatives
        # Enforce minimums
        if desired_pos < min_class_count or desired_neg < min_class_count:
            return idx_all
        pos_keep = rng.choice(pos, size=desired_pos, replace=False) if desired_pos < n_pos else pos
        neg_keep = neg
    else:
        # Need higher prevalence => downsample negatives
        desired_neg = min(n_neg, n_neg_req)
        desired_pos = n_pos  # keep all positives
        if desired_pos < min_class_count or desired_neg < min_class_count:
            return idx_all
        neg_keep = rng.choice(neg, size=desired_neg, replace=False) if desired_neg < n_neg else neg
        pos_keep = pos

    keep = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(keep)
    return keep

# ============================================================
# Outer split iterator (consistent with runner)
# ============================================================
def make_splits_from_runner_outer(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    split_key: str,
    seed: int,
):
    outer = get_outer(split_key, seed)
    if split_key == "S2":
        if groups is None:
            raise ValueError("split_key=S2 requires groups from load_dataset_A()")
        return outer, outer.split(X, y, groups)
    return outer, outer.split(X, y)

# ============================================================
# Core evaluation: one (split, seed, target_prev)
# ============================================================
def eval_one_setting(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    split_key: str,
    seed: int,
    target_prev: Optional[float],
    num_cols: List[str],
    cat_cols: List[str],
    *,
    show_tqdm: bool = True,
    pbar_pos: int = 0,
    desc_prefix: str = "",
) -> Dict[str, Any]:
    """
    Runs outer-CV once, evaluates each model using runner-consistent logic:
      - fold-safe preprocessor (P0 style)
      - calibration split BEFORE tuning
      - nested tuning on train_sub only
      - fit on train_sub, calibrate on cal_sub, evaluate on outer-test (unshifted)
    """
    fold_rows: List[Dict[str, Any]] = []

    outer, split_iter = make_splits_from_runner_outer(X, y, groups, split_key, seed)

    try:
        n_splits = outer.get_n_splits(X, y, groups) if split_key == "S2" else outer.get_n_splits(X, y)
    except Exception:
        n_splits = None

    fold_iter = enumerate(split_iter, start=1)
    if show_tqdm:
        fold_desc = f"{desc_prefix}outer folds".strip() or "outer folds"
        fold_iter = tqdm(
            fold_iter,
            total=n_splits,
            desc=fold_desc,
            position=pbar_pos,
            leave=False,
        )

    for fold_id, (tr_idx, te_idx) in fold_iter:
        X_tr_full = X.iloc[tr_idx].copy()
        y_tr_full = y[tr_idx]
        X_te = X.iloc[te_idx].copy()
        y_te = y[te_idx]
        g_tr_full = groups[tr_idx] if (split_key == "S2" and groups is not None) else None

        # Apply prevalence shift ONLY on outer-train fold (label-prior shift)
        if target_prev is not None:
            keep_rel = subsample_to_prevalence(y_tr_full, target_prev=float(target_prev), seed=seed + 1000 * fold_id)
            X_tr = X_tr_full.iloc[keep_rel].copy()
            y_tr = y_tr_full[keep_rel]
            g_tr = g_tr_full[keep_rel] if g_tr_full is not None else None
        else:
            X_tr = X_tr_full
            y_tr = y_tr_full
            g_tr = g_tr_full

        # Fold-safe preprocessor (P0)
        pre = build_preprocessor(
            num_cols=num_cols,
            cat_cols=cat_cols,
            include_imputer=True,
            include_scaler=True,
        )

        model_iter = MODELS
        if show_tqdm:
            model_iter = tqdm(
                MODELS,
                desc=f"{desc_prefix}fold {fold_id} models".strip() or f"fold {fold_id} models",
                position=pbar_pos + 1,
                leave=False,
            )

        # IMPORTANT: calibration split must be computed on (possibly shifted) y_tr/g_tr
        for model_key in model_iter:
            base_model, grid, do_cal = make_model_and_grid(model_key, seed)
            base_pipe = Pipeline(steps=[("pre", pre), ("clf", base_model)])

            tr_sub, cal_sub = calibration_split_indices(split_key, y_tr, g_tr, seed)

            X_tune = X_tr.iloc[tr_sub]
            y_tune = y_tr[tr_sub]
            g_tune = g_tr[tr_sub] if (split_key == "S2" and g_tr is not None) else None

            best_pipe, best_params = fit_best_model_nested(
                base_pipe=base_pipe,
                grid=grid,
                split_key=split_key,
                X_train=X_tune,
                y_train=y_tune,
                groups_train=g_tune,
                seed=seed,
            )

            best_pipe.fit(X_tune, y_tune)

            if do_cal:
                calibrator = PrefitCalibrator(best_pipe, method="sigmoid")
                calibrator.fit(X_tr.iloc[cal_sub], y_tr[cal_sub])
                final_model = calibrator
            else:
                final_model = best_pipe

            p = predict_proba_safe(final_model, X_te)

            auroc = roc_auc_score(y_te, p)
            ap = average_precision_score(y_te, p)
            brier = brier_score_loss(y_te, np.clip(p, 0.0, 1.0))

            fold_rows.append({
                "split": split_key,
                "seed": seed,
                "target_prev": (NATIVE_PREV_TAG if target_prev is None else float(target_prev)),
                "fold": fold_id,
                "model": model_key,
                "train_prev_effective": float(np.mean(y_tr)),
                "test_prev": float(np.mean(y_te)),
                "auroc": float(auroc),
                "ap": float(ap),
                "brier": float(brier),
                "best_params": json.dumps(best_params),
            })

    df = pd.DataFrame(fold_rows)

    agg = df.groupby(["split", "seed", "target_prev", "model"], as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        train_prev_effective=("train_prev_effective", "mean"),
        test_prev=("test_prev", "mean"),
    )
    for c in ["auroc_std", "ap_std", "brier_std"]:
        agg[c] = agg[c].fillna(0.0)

    agg_sorted = agg.sort_values(
        ["auroc_mean", "ap_mean", "brier_mean"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    winner_model = agg_sorted.iloc[0]["model"]
    winner_auc = float(agg_sorted.iloc[0]["auroc_mean"])
    winner_ap = float(agg_sorted.iloc[0]["ap_mean"])
    winner_brier = float(agg_sorted.iloc[0]["brier_mean"])
    ranking = agg_sorted["model"].tolist()

    return {
        "fold_metrics": df,
        "model_summary": agg,
        "winner_model": winner_model,
        "winner_auc": winner_auc,
        "winner_ap": winner_ap,
        "winner_brier": winner_brier,
        "ranking": ranking,
    }

# ============================================================
# Robustness envelope + flip onset (same semantics as Phase4a)
# ============================================================
@dataclass
class EnvelopePoint:
    target_prev: Optional[float]
    winner_flip_pct: float
    kendall_tau_mean: float
    spearman_rho_mean: float
    baseline_winner: str
    stable: bool

def compute_stability_against_baseline(
    rankings_by_seed: Dict[int, List[str]],
    baseline_seed: int,
) -> Tuple[float, float]:
    base_rank = rankings_by_seed[baseline_seed]
    base_pos = {m: i for i, m in enumerate(base_rank)}

    taus, rhos = [], []
    for s, rank in rankings_by_seed.items():
        if s == baseline_seed:
            continue
        pos = {m: i for i, m in enumerate(rank)}
        common = [m for m in base_rank if m in pos]
        x = [base_pos[m] for m in common]
        y = [pos[m] for m in common]
        tau = kendalltau(x, y).correlation
        rho = spearmanr(x, y).correlation
        taus.append(float(tau) if tau == tau else 0.0)
        rhos.append(float(rho) if rho == rho else 0.0)

    return float(np.mean(taus)), float(np.mean(rhos))

def build_envelope(
    winners_df: pd.DataFrame,
    rankings_map: Dict[Tuple[str, Optional[float]], Dict[int, List[str]]],
    baseline_seed: int,
    split_key: str,
) -> List[EnvelopePoint]:
    points: List[EnvelopePoint] = []
    prevs = winners_df["target_prev"].unique().tolist()

    # Ensure deterministic ordering: baseline (None) first, then numeric ascending
    prevs_sorted: List[Optional[float]] = [None] + sorted([p for p in prevs if p is not None and p != NATIVE_PREV_TAG])

    base_winner = winners_df[
        (winners_df["split"] == split_key) &
        (winners_df["target_prev"] == NATIVE_PREV_TAG) &
        (winners_df["seed"] == baseline_seed)
    ]["winner_model"].iloc[0]

    for tp in prevs_sorted:
        if tp is None:
            sub = winners_df[(winners_df["split"] == split_key) & (winners_df["target_prev"] == NATIVE_PREV_TAG)]
        else:
            sub = winners_df[(winners_df["split"] == split_key) & (winners_df["target_prev"] == tp)]

        flip_pct = 100.0 * (sub["winner_model"] != base_winner).mean()

        rankings_by_seed = rankings_map[(split_key, tp)]
        tau_mean, rho_mean = compute_stability_against_baseline(rankings_by_seed, baseline_seed)

        stable = (flip_pct <= MAX_WINNER_FLIP_PCT) and (tau_mean >= MIN_KENDALL_TAU)

        points.append(EnvelopePoint(
            target_prev=tp,
            winner_flip_pct=float(flip_pct),
            kendall_tau_mean=float(tau_mean),
            spearman_rho_mean=float(rho_mean),
            baseline_winner=base_winner,
            stable=bool(stable),
        ))

    return points

# ============================================================
# MAIN SWEEP
# ============================================================
def main():
    X, y, groups, num_cols, cat_cols = load_dataset_A()

    all_fold_metrics = []
    all_model_summaries = []
    winners_rows = []
    rankings_map: Dict[Tuple[str, Optional[float]], Dict[int, List[str]]] = {}

    split_iter = tqdm(SPLITS, desc="Split policy", position=0, leave=True)
    for split_key in split_iter:
        tp_iter = tqdm(TARGET_PREVS, desc=f"{split_key}: target prev", position=1, leave=False)
        for tp in tp_iter:
            rankings_map[(split_key, tp)] = {}

            seed_iter = tqdm(SEEDS, desc=f"{split_key} tp={tp}: seeds", position=2, leave=False)
            for seed in seed_iter:
                out = eval_one_setting(
                    X=X, y=y, groups=groups,
                    split_key=split_key, seed=seed, target_prev=tp,
                    num_cols=num_cols, cat_cols=cat_cols,
                    show_tqdm=True,
                    pbar_pos=3,
                    desc_prefix=f"{split_key} tp={tp} s={seed} | ",
                )

                all_fold_metrics.append(out["fold_metrics"])
                all_model_summaries.append(out["model_summary"])

                winners_rows.append({
                    "split": split_key,
                    "seed": seed,
                    "target_prev": (NATIVE_PREV_TAG if tp is None else float(tp)),
                    "winner_model": out["winner_model"],
                    "winner_auc": out["winner_auc"],
                    "winner_ap": out["winner_ap"],
                    "winner_brier": out["winner_brier"],
                })

                rankings_map[(split_key, tp)][seed] = out["ranking"]

                tqdm.write(
                    f"[OK] split={split_key} tp={tp} seed={seed} "
                    f"winner={out['winner_model']} auc={out['winner_auc']:.4f} "
                    f"ap={out['winner_ap']:.4f} brier={out['winner_brier']:.4f}"
                )

    fold_df = pd.concat(all_fold_metrics, ignore_index=True)
    summary_df = pd.concat(all_model_summaries, ignore_index=True)
    winners_df = pd.DataFrame(winners_rows)

    fold_df.to_csv(os.path.join(OUT_DIR, "prevalence_shift_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "prevalence_shift_summary_by_model.csv"), index=False)
    winners_df.to_csv(os.path.join(OUT_DIR, "prevalence_shift_winner_by_seed.csv"), index=False)

    baseline_seed = SEEDS[0]

    envelope_rows = []
    flip_onset_rows = []
    for split_key in SPLITS:
        points = build_envelope(winners_df, rankings_map, baseline_seed=baseline_seed, split_key=split_key)

        # flip onset: first non-baseline target_prev where flip_pct > 0
        onset = None
        for p in points:
            if p.target_prev is None:
                continue
            if p.winner_flip_pct > 0.0:
                onset = p.target_prev
                break

        flip_onset_rows.append({
            "split": split_key,
            "baseline_seed": baseline_seed,
            "baseline_winner": points[0].baseline_winner,
            "flip_begins_at_target_prev": onset,
        })

        for p in points:
            envelope_rows.append({
                "split": split_key,
                "target_prev": (NATIVE_PREV_TAG if p.target_prev is None else float(p.target_prev)),
                "baseline_seed": baseline_seed,
                "baseline_winner": p.baseline_winner,
                "winner_flip_pct": p.winner_flip_pct,
                "kendall_tau_mean": p.kendall_tau_mean,
                "spearman_rho_mean": p.spearman_rho_mean,
                "stable_under_thresholds": p.stable,
            })

    env_df = pd.DataFrame(envelope_rows)
    onset_df = pd.DataFrame(flip_onset_rows)

    env_df.to_csv(os.path.join(OUT_DIR, "robustness_envelope_prevalence_shift.csv"), index=False)
    onset_df.to_csv(os.path.join(OUT_DIR, "flip_onset_prevalence_shift.csv"), index=False)

    tqdm.write("\n=== DONE (PHASE 4b) ===")
    tqdm.write(f"Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
