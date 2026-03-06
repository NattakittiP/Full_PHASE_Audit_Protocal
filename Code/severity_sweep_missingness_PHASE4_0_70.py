#  
# ------------------------------------------------------------
# Perturbation severity sweep (MCAR missingness 5/10/20/30%)
# - Leakage-controlled evaluation (fold-safe imputation/scaling)
# - Show when winner flip begins
# - Define "robustness envelope" (max missing rate before instability)
#
# PATCH: Make Phase 4 winner selection CONSISTENT with Phase 1–3:
#   Winner ranking key = (AUROC desc, AP desc, Brier asc)  [same as runner rank_key]
#   -> adds Brier computation per fold, aggregates brier_mean, and uses the same rank_key.
# ------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import kendalltau, spearmanr

# ============================================================
# IMPORT SECTION (runner functions)  <<<< IMPORTANT
# ============================================================
from jcsse_audit_runner_tqdm_hardened import (  # must match your attached runner filename
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
OUT_DIR = "PHASE4_SEVERITY_SWEEP"
os.makedirs(OUT_DIR, exist_ok=True)

MISS_RATES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]   # 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%
SEEDS = list(range(1001, 1021))         # 20 seeds
SPLITS = ["S1", "S2"]

# IMPORTANT: keep model set consistent with runner
MODELS = ["lr_l2", "svm_linear_cal", "rf", "xgb", "extratrees"]

# Stability thresholds for "robustness envelope"
MAX_WINNER_FLIP_PCT = 5.0
MIN_KENDALL_TAU = 0.8

# ============================================================
# Phase 1–3 consistent winner ranking key
# rank by AUROC desc, then AP desc, then Brier asc
# (same semantics as runner's rank_key(mean_auc, mean_ap, mean_brier) -> (auc, ap, -brier))
# ============================================================
def rank_key(mean_auc: float, mean_ap: float, mean_brier: float) -> Tuple[float, float, float]:
    return (float(mean_auc), float(mean_ap), -float(mean_brier))

# ============================================================
# Missingness injection (MCAR) – leakage-safe design
# ============================================================
def apply_mcar_missingness_split(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    rate: float,
    seed: int,
    num_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply MCAR missingness to NUMERIC columns only (recommended).
    Train/test masked independently (separate RNG) to avoid coupling.
    """
    X_tr2 = X_tr.copy()
    X_te2 = X_te.copy()

    rng_tr = np.random.default_rng(seed + 12345 + int(rate * 1000))
    rng_te = np.random.default_rng(seed + 54321 + int(rate * 1000))

    for c in num_cols:
        if c not in X_tr2.columns:
            continue
        m_tr = rng_tr.random(len(X_tr2)) < rate
        m_te = rng_te.random(len(X_te2)) < rate
        X_tr2.loc[m_tr, c] = np.nan
        X_te2.loc[m_te, c] = np.nan

    return X_tr2, X_te2

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
    """
    Create the runner-consistent outer CV splitter and its iterator.
    """
    outer = get_outer(split_key, seed)
    if split_key == "S2":
        if groups is None:
            raise ValueError("split_key=S2 requires groups (subject_id) from load_dataset_A()")
        return outer, outer.split(X, y, groups)
    return outer, outer.split(X, y)

# ============================================================
# Core evaluation: one (split, seed, miss_rate)
# ============================================================
def eval_one_setting(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    split_key: str,
    seed: int,
    miss_rate: float,
    num_cols: List[str],
    cat_cols: List[str],
    *,
    show_tqdm: bool = True,
    pbar_pos: int = 0,
    desc_prefix: str = "",
) -> Dict:
    """
    Runs outer-CV once, evaluates each model using runner-consistent logic:
      - fold-safe preprocessor (P0 style)
      - nested tuning via fit_best_model_nested()
      - calibration inside outer-train using PrefitCalibrator when do_cal=True

    PATCH: Also compute Brier per fold and aggregate Brier mean.
           Winner selection + ranking use (AUROC, AP, -Brier) like Phase 1–3.
    """
    fold_rows = []

    outer, split_iter = make_splits_from_runner_outer(X, y, groups, split_key, seed)

    # Best-effort fold count for tqdm (handles group-aware splitters too)
    try:
        n_splits = outer.get_n_splits(X, y, groups) if split_key == "S2" else outer.get_n_splits(X, y)
    except Exception:
        n_splits = None

    fold_iter = enumerate(split_iter, start=1)
    if show_tqdm:
        fold_desc = f"{desc_prefix}outer folds".strip()
        fold_iter = tqdm(
            fold_iter,
            total=n_splits,
            desc=fold_desc if fold_desc else "outer folds",
            position=pbar_pos,
            leave=False,
        )

    from sklearn.pipeline import Pipeline

    for fold_id, (tr_idx, te_idx) in fold_iter:
        X_tr = X.iloc[tr_idx].copy()
        X_te = X.iloc[te_idx].copy()
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        g_tr = groups[tr_idx] if (split_key == "S2" and groups is not None) else None

        # Apply MCAR missingness (label-free perturbation)
        if miss_rate > 0:
            X_tr, X_te = apply_mcar_missingness_split(
                X_tr, X_te, miss_rate, seed=seed + fold_id, num_cols=num_cols
            )

        # Build fold-safe preprocessor (P0 = include_imputer + include_scaler)
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
                desc=f"{desc_prefix}fold {fold_id} models".strip(),
                position=pbar_pos + 1,
                leave=False,
            )

        for model_key in model_iter:
            base_model, grid, do_cal = make_model_and_grid(model_key, seed)
            base_pipe = Pipeline(steps=[("pre", pre), ("clf", base_model)])

            # --------------------------------------------------------
            # (Leakage-hardening) calibration split happens BEFORE tuning
            # so that cal_sub is NEVER "seen" during hyperparameter search.
            # --------------------------------------------------------
            tr_sub, cal_sub = calibration_split_indices(split_key, y_tr, g_tr, seed)

            X_tune = X_tr.iloc[tr_sub]
            y_tune = y_tr[tr_sub]
            g_tune = g_tr[tr_sub] if (split_key == "S2" and g_tr is not None) else None

            # Nested tuning on train_sub only (runner function)
            best_pipe, best_params = fit_best_model_nested(
                base_pipe=base_pipe,
                grid=grid,
                split_key=split_key,
                X_train=X_tune,
                y_train=y_tune,
                groups_train=g_tune,
                seed=seed,
            )

            # Fit best model on train_sub
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
                "miss_rate": miss_rate,
                "fold": fold_id,
                "model": model_key,
                "auroc": float(auroc),
                "ap": float(ap),
                "brier": float(brier),
                "best_params": json.dumps(best_params),
            })

    df = pd.DataFrame(fold_rows)

    # Aggregate per model across folds
    agg = df.groupby(["split", "seed", "miss_rate", "model"], as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
    )
    for c in ["auroc_std", "ap_std", "brier_std"]:
        agg[c] = agg[c].fillna(0.0)

    # Determine winner + full ranking using Phase 1–3 consistent rule
    # Sort by AUROC desc, then AP desc, then Brier asc
    agg_sorted = agg.sort_values(
        ["auroc_mean", "ap_mean", "brier_mean"],
        ascending=[False, False, True],
        kind="mergesort",  # stable sort (nice for reproducibility)
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
# Robustness envelope + flip onset
# ============================================================
@dataclass
class EnvelopePoint:
    miss_rate: float
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
    rankings_map: Dict[Tuple[str, float], Dict[int, List[str]]],
    baseline_seed: int,
    split_key: str,
) -> List[EnvelopePoint]:
    points = []
    rates = sorted(winners_df["miss_rate"].unique().tolist())
    base_rate = 0.0 if 0.0 in rates else rates[0]

    base_winner = winners_df[
        (winners_df["split"] == split_key) &
        (winners_df["miss_rate"] == base_rate) &
        (winners_df["seed"] == baseline_seed)
    ]["winner_model"].iloc[0]

    for r in rates:
        sub = winners_df[(winners_df["split"] == split_key) & (winners_df["miss_rate"] == r)]
        flip_pct = 100.0 * (sub["winner_model"] != base_winner).mean()

        rankings_by_seed = rankings_map[(split_key, r)]
        tau_mean, rho_mean = compute_stability_against_baseline(rankings_by_seed, baseline_seed)

        stable = (flip_pct <= MAX_WINNER_FLIP_PCT) and (tau_mean >= MIN_KENDALL_TAU)
        points.append(EnvelopePoint(
            miss_rate=r,
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
    # Use runner's Dataset A loader (already normalizes + drops hadm_id/subject_id)
    X, y, groups, num_cols, cat_cols = load_dataset_A()

    all_rates = [0.0] + MISS_RATES

    all_fold_metrics = []
    all_model_summaries = []
    winners_rows = []
    rankings_map: Dict[Tuple[str, float], Dict[int, List[str]]] = {}

    split_iter = tqdm(SPLITS, desc="Split policy", position=0, leave=True)

    for split_key in split_iter:
        rate_iter = tqdm(all_rates, desc=f"{split_key}: missingness rates", position=1, leave=False)
        for miss_rate in rate_iter:
            rankings_map[(split_key, miss_rate)] = {}

            seed_iter = tqdm(SEEDS, desc=f"{split_key} rate={miss_rate:.2f}: seeds", position=2, leave=False)
            for seed in seed_iter:
                out = eval_one_setting(
                    X=X, y=y, groups=groups,
                    split_key=split_key, seed=seed, miss_rate=miss_rate,
                    num_cols=num_cols, cat_cols=cat_cols,
                    show_tqdm=True,
                    pbar_pos=3,
                    desc_prefix=f"{split_key} r={miss_rate:.2f} s={seed} | ",
                )

                all_fold_metrics.append(out["fold_metrics"])
                all_model_summaries.append(out["model_summary"])

                winners_rows.append({
                    "split": split_key,
                    "seed": seed,
                    "miss_rate": miss_rate,
                    "winner_model": out["winner_model"],
                    "winner_auc": out["winner_auc"],
                    "winner_ap": out.get("winner_ap", np.nan),
                    "winner_brier": out.get("winner_brier", np.nan),
                })

                rankings_map[(split_key, miss_rate)][seed] = out["ranking"]

                tqdm.write(
                    f"[OK] split={split_key} rate={miss_rate:.2f} seed={seed} "
                    f"winner={out['winner_model']} auc={out['winner_auc']:.4f} "
                    f"ap={out.get('winner_ap', float('nan')):.4f} brier={out.get('winner_brier', float('nan')):.4f}"
                )

    fold_df = pd.concat(all_fold_metrics, ignore_index=True)
    summary_df = pd.concat(all_model_summaries, ignore_index=True)
    winners_df = pd.DataFrame(winners_rows)

    fold_df.to_csv(os.path.join(OUT_DIR, "severity_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "severity_summary_by_model.csv"), index=False)
    winners_df.to_csv(os.path.join(OUT_DIR, "severity_winner_by_seed.csv"), index=False)

    baseline_seed = SEEDS[0]

    envelope_rows = []
    flip_onset_rows = []
    for split_key in SPLITS:
        points = build_envelope(winners_df, rankings_map, baseline_seed=baseline_seed, split_key=split_key)

        onset = None
        for p in points:
            if p.miss_rate == 0.0:
                continue
            if p.winner_flip_pct > 0.0:
                onset = p.miss_rate
                break

        flip_onset_rows.append({
            "split": split_key,
            "baseline_seed": baseline_seed,
            "baseline_winner": points[0].baseline_winner,
            "flip_begins_at_miss_rate": onset,
        })

        for p in points:
            envelope_rows.append({
                "split": split_key,
                "miss_rate": p.miss_rate,
                "baseline_seed": baseline_seed,
                "baseline_winner": p.baseline_winner,
                "winner_flip_pct": p.winner_flip_pct,
                "kendall_tau_mean": p.kendall_tau_mean,
                "spearman_rho_mean": p.spearman_rho_mean,
                "stable_under_thresholds": p.stable,
            })

    env_df = pd.DataFrame(envelope_rows)
    onset_df = pd.DataFrame(flip_onset_rows)

    env_df.to_csv(os.path.join(OUT_DIR, "robustness_envelope.csv"), index=False)
    onset_df.to_csv(os.path.join(OUT_DIR, "flip_onset.csv"), index=False)

    tqdm.write("\n=== DONE ===")
    tqdm.write(f"Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()