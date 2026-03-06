# ------------------------------------------------------------
# PHASE 5 — Tie-aware ranking + Envelope sensitivity + Margin analysis
#
# Purpose (Q1-hardening / reviewer-proof):
#   1) Tie-aware model selection:
#        - Treat near-ties in AUROC/AP/Brier as ties (epsilon thresholds)
#        - Report co-winner sets and "tie-aware flip%" (baseline winner not in co-winner set)
#   2) Envelope sensitivity analysis:
#        - Sweep MAX_WINNER_FLIP_PCT ∈ {0, 5, 10}
#        - Sweep MIN_KENDALL_TAU ∈ {0.7, 0.8, 0.9}
#        - Report stable max severity and flip onset under each setting
#   3) Margin analysis:
#        - For each (split, severity, seed), compute margins between rank1 and rank2
#          in AUROC/AP/Brier to explain *why* flips happen (near-ties vs true reversals).
#
# Inputs:
#   - Phase4a outputs (missingness): directory created by severity_sweep_missingness_PHASE_0_70.py
#       PHASE4_SEVERITY_SWEEP/
#         severity_summary_by_model.csv
#         severity_winner_by_seed.csv   (optional; Phase 5 recomputes winners)
#
#   - Phase4b outputs (prevalence shift): directory created by PHASE 4b script
#       PHASE4B_PREVALENCE_SHIFT/
#         prevalence_shift_summary_by_model.csv
#         prevalence_shift_winner_by_seed.csv (optional)
#
# Outputs:
#   PHASE5_ANALYSIS/
#     phase5_winners_tieaware_missingness.csv
#     phase5_winners_tieaware_prevalence_shift.csv
#     phase5_envelope_sensitivity_missingness.csv
#     phase5_envelope_sensitivity_prevalence_shift.csv
#     phase5_margins_missingness.csv
#     phase5_margins_prevalence_shift.csv
#
# Notes:
#   - This phase is *postprocessing only* (no model training),
#     but it is 100% consistent with Phase1–4 winner rule:
#       AUROC desc, AP desc, Brier asc
#     and then adds tie-aware "equivalence" thresholds.
# ------------------------------------------------------------

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import kendalltau, spearmanr

# ============================================================
# CONFIG
# ============================================================
PHASE4_DIR = "PHASE4_SEVERITY_SWEEP"
PHASE4B_DIR = "PHASE4B_PREVALENCE_SHIFT"
OUT_DIR = "PHASE5_ANALYSIS"
os.makedirs(OUT_DIR, exist_ok=True)

# Tie thresholds (can tweak if reviewer asks; keep small and interpretable)
EPS_AUROC = 1e-3
EPS_AP = 1e-3
EPS_BRIER = 1e-4  # brier is on ~[0,0.25], so tighter epsilon is reasonable

# Envelope sensitivity sweep
FLIP_PCT_GRID = [0.0, 5.0, 10.0]
TAU_GRID = [0.7, 0.8, 0.9]

# ============================================================
# Helpers: deterministic winner and tie-aware co-winner set
# ============================================================
def deterministic_rank(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic ranking: AUROC desc, AP desc, Brier asc.
    Expects columns: auroc_mean, ap_mean, brier_mean, model
    """
    return df_sub.sort_values(
        ["auroc_mean", "ap_mean", "brier_mean"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

def tie_aware_winners(df_sub: pd.DataFrame,
                      eps_auc: float = EPS_AUROC,
                      eps_ap: float = EPS_AP,
                      eps_brier: float = EPS_BRIER) -> List[str]:
    """
    Return a list of co-winners under tie-aware lexicographic rule:
      1) Keep all models within eps_auc of best AUROC
      2) Among those, keep within eps_ap of best AP
      3) Among those, keep within eps_brier of best (lowest) Brier
    """
    s = df_sub.copy()
    if len(s) == 0:
        return []

    # Step 1: AUROC
    best_auc = s["auroc_mean"].max()
    s = s[s["auroc_mean"] >= best_auc - eps_auc].copy()
    if len(s) == 1:
        return s["model"].tolist()

    # Step 2: AP
    best_ap = s["ap_mean"].max()
    s = s[s["ap_mean"] >= best_ap - eps_ap].copy()
    if len(s) == 1:
        return s["model"].tolist()

    # Step 3: Brier (lower better)
    best_brier = s["brier_mean"].min()
    s = s[s["brier_mean"] <= best_brier + eps_brier].copy()

    # Stable ordering for reporting
    s = deterministic_rank(s)
    return s["model"].tolist()

# ============================================================
# Ranking similarity vs baseline seed
# ============================================================
def kendall_spearman_against_baseline(rankings_by_seed: Dict[int, List[str]], baseline_seed: int) -> Tuple[float, float]:
    base_rank = rankings_by_seed[baseline_seed]
    base_pos = {m: i for i, m in enumerate(base_rank)}
    taus, rhos = [], []
    for seed, rank in rankings_by_seed.items():
        if seed == baseline_seed:
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

# ============================================================
# Core postprocess for a stress type
# ============================================================
@dataclass
class StressSpec:
    name: str
    path: str
    summary_file: str
    severity_col: str  # e.g., "miss_rate" or "target_prev"

def load_stress_summary(spec: StressSpec) -> pd.DataFrame:
    p = os.path.join(spec.path, spec.summary_file)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")
    df = pd.read_csv(p)

    # Normalize severity: ensure baseline severity exists
    if spec.severity_col not in df.columns:
        raise ValueError(f"Expected column '{spec.severity_col}' in {p}")
    return df

def compute_winners_and_rankings(df: pd.DataFrame, severity_col: str) -> Tuple[pd.DataFrame, Dict[Tuple[str, float], Dict[int, List[str]]], Dict[Tuple[str, float], Dict[int, List[str]]]]:
    """
    From per-seed per-model summary, compute:
      - deterministic winner per (split, seed, severity)
      - tie-aware co-winners per (split, seed, severity)
      - deterministic full ranking per (split, seed, severity) for tau/rho
    Returns:
      winners_df, rankings_det, winnerset_tie
    """
    required = {"split", "seed", severity_col, "model", "auroc_mean", "ap_mean", "brier_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary missing columns: {sorted(missing)}")

    winners_rows = []
    rankings_det: Dict[Tuple[str, float], Dict[int, List[str]]] = {}
    winnerset_tie: Dict[Tuple[str, float], Dict[int, List[str]]] = {}

    # Ensure numeric
    df = df.copy()
    df["seed"] = df["seed"].astype(int)
    df[severity_col] = df[severity_col].astype(float)

    groups = df.groupby(["split", severity_col, "seed"], sort=True)
    for (split, sev, seed), sub in tqdm(groups, desc="Compute winners/rankings", leave=False):
        sub_ranked = deterministic_rank(sub)
        winner = sub_ranked.iloc[0]["model"]

        co_winners = tie_aware_winners(sub_ranked)

        winners_rows.append({
            "split": split,
            "seed": int(seed),
            severity_col: float(sev),
            "winner_det": winner,
            "winner_tie_set": "|".join(co_winners),
            "winner_tie_k": int(len(co_winners)),
            "best_auroc": float(sub_ranked.iloc[0]["auroc_mean"]),
            "best_ap": float(sub_ranked.iloc[0]["ap_mean"]),
            "best_brier": float(sub_ranked.iloc[0]["brier_mean"]),
        })

        rankings_det.setdefault((split, float(sev)), {})[int(seed)] = sub_ranked["model"].tolist()
        winnerset_tie.setdefault((split, float(sev)), {})[int(seed)] = co_winners

    winners_df = pd.DataFrame(winners_rows)
    return winners_df, rankings_det, winnerset_tie

def compute_margins(df: pd.DataFrame, severity_col: str) -> pd.DataFrame:
    """
    For each (split, severity, seed), compute margins between rank1 and rank2:
      margin_auc = auc1 - auc2
      margin_ap  = ap1  - ap2
      margin_brier = brier2 - brier1  (positive means winner has better/lower brier)
    """
    out_rows = []
    df = df.copy()
    df["seed"] = df["seed"].astype(int)
    df[severity_col] = df[severity_col].astype(float)

    for (split, sev, seed), sub in tqdm(df.groupby(["split", severity_col, "seed"]), desc="Compute margins", leave=False):
        sub_ranked = deterministic_rank(sub)
        if len(sub_ranked) < 2:
            continue
        r1 = sub_ranked.iloc[0]
        r2 = sub_ranked.iloc[1]
        out_rows.append({
            "split": split,
            "seed": int(seed),
            severity_col: float(sev),
            "rank1_model": r1["model"],
            "rank2_model": r2["model"],
            "margin_auc": float(r1["auroc_mean"] - r2["auroc_mean"]),
            "margin_ap": float(r1["ap_mean"] - r2["ap_mean"]),
            "margin_brier": float(r2["brier_mean"] - r1["brier_mean"]),
        })
    return pd.DataFrame(out_rows)

def envelope_sensitivity(
    winners_df: pd.DataFrame,
    rankings_det: Dict[Tuple[str, float], Dict[int, List[str]]],
    winnerset_tie: Dict[Tuple[str, float], Dict[int, List[str]]],
    severity_col: str,
    baseline_seed: int,
    *,
    flip_grid: List[float] = FLIP_PCT_GRID,
    tau_grid: List[float] = TAU_GRID,
) -> pd.DataFrame:
    """
    Compute envelope metrics under threshold sweeps for BOTH:
      - deterministic flips (winner_det != baseline_det)
      - tie-aware flips (baseline_det not in tie set)
    """
    rows = []
    splits = sorted(winners_df["split"].unique().tolist())

    # Determine baseline severity per split: smallest numeric value
    # For missingness, that's 0.0. For prevalence shift, baseline is NaN in raw file,
    # but Phase4b summary writes NaN; we rely on the fact it will be converted to float and likely NaN.
    # Here we handle by using min severity among non-NaN, and allow user to override by pre-filling baseline.
    for split in splits:
        sub_split = winners_df[winners_df["split"] == split].copy()

        # Baseline severity: minimum value
        sev_vals = sorted(sub_split[severity_col].dropna().unique().tolist())
        if len(sev_vals) == 0:
            continue
        baseline_sev = float(min(sev_vals))

        # Baseline deterministic winner (at baseline_sev, baseline_seed)
        base_row = sub_split[(sub_split[severity_col] == baseline_sev) & (sub_split["seed"] == baseline_seed)]
        if len(base_row) == 0:
            # fallback: first seed at baseline severity
            base_row = sub_split[sub_split[severity_col] == baseline_sev].sort_values("seed").head(1)
        baseline_det = base_row.iloc[0]["winner_det"]

        # Precompute flip% and tau/rho at each severity
        for sev in sev_vals:
            sub_sev = sub_split[sub_split[severity_col] == sev].copy()

            det_flip_pct = 100.0 * (sub_sev["winner_det"] != baseline_det).mean()

            # Tie-aware flip: baseline_det not in tie set
            def is_flip_tie(seed: int) -> int:
                tie_set = winnerset_tie[(split, float(sev))][int(seed)]
                return int(baseline_det not in tie_set)
            tie_flips = [is_flip_tie(int(s)) for s in sub_sev["seed"].tolist()]
            tie_flip_pct = 100.0 * float(np.mean(tie_flips)) if len(tie_flips) else 0.0

            tau_mean, rho_mean = kendall_spearman_against_baseline(rankings_det[(split, float(sev))], baseline_seed)

            for flip_thr in flip_grid:
                for tau_thr in tau_grid:
                    stable_det = (det_flip_pct <= flip_thr) and (tau_mean >= tau_thr)
                    stable_tie = (tie_flip_pct <= flip_thr) and (tau_mean >= tau_thr)

                    rows.append({
                        "split": split,
                        severity_col: float(sev),
                        "baseline_seed": int(baseline_seed),
                        "baseline_severity": float(baseline_sev),
                        "baseline_winner_det": baseline_det,
                        "det_flip_pct": float(det_flip_pct),
                        "tie_flip_pct": float(tie_flip_pct),
                        "kendall_tau_mean": float(tau_mean),
                        "spearman_rho_mean": float(rho_mean),
                        "flip_thr": float(flip_thr),
                        "tau_thr": float(tau_thr),
                        "stable_det": bool(stable_det),
                        "stable_tie": bool(stable_tie),
                    })

    df_out = pd.DataFrame(rows)

    # Also compute "max stable severity" and "flip onset" per threshold for convenience
    # (keep as separate pass, but stored in the same table via additional rows)
    summary_rows = []
    for (split, flip_thr, tau_thr), sub in df_out.groupby(["split", "flip_thr", "tau_thr"], sort=True):
        sev_sorted = sub.sort_values(severity_col)
        # Max stable severity
        max_stable_det = sev_sorted[sev_sorted["stable_det"]][severity_col].max()
        max_stable_tie = sev_sorted[sev_sorted["stable_tie"]][severity_col].max()

        # Flip onset: first severity > baseline_severity with flip_pct > 0
        baseline_sev = float(sev_sorted["baseline_severity"].iloc[0])
        onset_det = None
        onset_tie = None
        for _, r in sev_sorted.iterrows():
            if float(r[severity_col]) <= baseline_sev:
                continue
            if onset_det is None and r["det_flip_pct"] > 0.0:
                onset_det = float(r[severity_col])
            if onset_tie is None and r["tie_flip_pct"] > 0.0:
                onset_tie = float(r[severity_col])
        summary_rows.append({
            "split": split,
            "flip_thr": float(flip_thr),
            "tau_thr": float(tau_thr),
            "baseline_seed": int(sev_sorted["baseline_seed"].iloc[0]),
            "baseline_severity": baseline_sev,
            "baseline_winner_det": sev_sorted["baseline_winner_det"].iloc[0],
            "max_stable_severity_det": (np.nan if max_stable_det != max_stable_det else float(max_stable_det)),
            "max_stable_severity_tie": (np.nan if max_stable_tie != max_stable_tie else float(max_stable_tie)),
            "flip_onset_det": onset_det,
            "flip_onset_tie": onset_tie,
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_out, df_summary

# ============================================================
# MAIN
# ============================================================
def run_one(spec: StressSpec, out_prefix: str):
    df = load_stress_summary(spec)


    # Defensive: treat NaN target_prev (native) as -1.0 for stable numeric handling
    if spec.severity_col == "target_prev" and df[spec.severity_col].isna().any():
        df[spec.severity_col] = df[spec.severity_col].fillna(-1.0)
    winners_df, rankings_det, winnerset_tie = compute_winners_and_rankings(df, spec.severity_col)
    margins_df = compute_margins(df, spec.severity_col)

    # Baseline seed: use smallest seed present
    baseline_seed = int(winners_df["seed"].min())

    env_detail, env_summary = envelope_sensitivity(
        winners_df, rankings_det, winnerset_tie, spec.severity_col, baseline_seed=baseline_seed
    )

    winners_df.to_csv(os.path.join(OUT_DIR, f"phase5_winners_tieaware_{out_prefix}.csv"), index=False)
    margins_df.to_csv(os.path.join(OUT_DIR, f"phase5_margins_{out_prefix}.csv"), index=False)
    env_detail.to_csv(os.path.join(OUT_DIR, f"phase5_envelope_sensitivity_{out_prefix}.csv"), index=False)
    env_summary.to_csv(os.path.join(OUT_DIR, f"phase5_envelope_summary_{out_prefix}.csv"), index=False)

    print(f"[OK] Phase5 outputs written for {spec.name} -> {OUT_DIR}")

def main():
    # Phase4a (missingness)
    spec4 = StressSpec(
        name="missingness",
        path=PHASE4_DIR,
        summary_file="severity_summary_by_model.csv",
        severity_col="miss_rate",
    )

    # Phase4b (prevalence shift)
    spec4b = StressSpec(
        name="prevalence_shift",
        path=PHASE4B_DIR,
        summary_file="prevalence_shift_summary_by_model.csv",
        severity_col="target_prev",
    )

    run_one(spec4, out_prefix="missingness")
    run_one(spec4b, out_prefix="prevalence_shift")

    print("\n=== DONE (PHASE 5) ===")

if __name__ == "__main__":
    main()
