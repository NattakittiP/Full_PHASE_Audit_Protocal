# ============================================================
# Fig_Generate.py — Paper-quality figure generator (Phase 5)
# (Adjusted to match the attached input CSV schemas exactly)
#
# INPUT (as attached):
#   phase5_envelope_sensitivity_missingness.csv
#   phase5_envelope_sensitivity_prevalence_shift.csv
#   phase5_margins_missingness.csv
#   phase5_margins_prevalence_shift.csv
#   phase5_winners_tieaware_missingness.csv
#   phase5_winners_tieaware_prevalence_shift.csv
#
# Key schema fixes:
#   - flip_threshold   -> flip_thr
#   - tau_threshold    -> tau_thr
#   - stable_under_thresholds -> stable_det / stable_tie (boolean)
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Plot Style (STRICTLY as requested)
# ============================================================
FIGSIZE = (6.5, 4.8)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

def set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.3,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })

# ============================================================
# Helpers
# ============================================================
def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)

def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _assert_columns(df: pd.DataFrame, needed: list, name: str):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

# ============================================================
# Figure 1/2: Flip curve + Stability curve (Missingness)
# ============================================================
def fig_missingness_flip_curve(env: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(env, ["split", "miss_rate", "det_flip_pct", "tie_flip_pct"],
                    "phase5_envelope_sensitivity_missingness.csv")
    env = env.copy()
    env = _ensure_numeric(env, ["miss_rate", "det_flip_pct", "tie_flip_pct"])

    g = env.groupby(["split", "miss_rate"], as_index=False).agg(
        det_flip=("det_flip_pct", "mean"),
        tie_flip=("tie_flip_pct", "mean"),
    ).sort_values(["split", "miss_rate"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for split in sorted(g["split"].unique()):
        sub = g[g["split"] == split].sort_values("miss_rate")
        ax.plot(sub["miss_rate"], sub["det_flip"], marker="o", label=f"{split} (deterministic)")
        ax.plot(sub["miss_rate"], sub["tie_flip"], marker="s", linestyle="--", label=f"{split} (tie-aware)")

    ax.set_xlabel("MCAR Missingness Rate")
    ax.set_ylabel("Winner Flip (%)")
    ax.set_xlim(0, float(np.nanmax(g["miss_rate"])) if len(g) else 0.7)
    ax.set_ylim(0)
    ax.grid(True)
    ax.legend()
    _save(fig, out_dir, "Fig1_flip_curve_missingness.png")

def fig_missingness_stability_curve(env: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(env, ["split", "miss_rate", "kendall_tau_mean", "spearman_rho_mean"],
                    "phase5_envelope_sensitivity_missingness.csv")
    env = env.copy()
    env = _ensure_numeric(env, ["miss_rate", "kendall_tau_mean", "spearman_rho_mean"])

    g = env.groupby(["split", "miss_rate"], as_index=False).agg(
        tau=("kendall_tau_mean", "mean"),
        rho=("spearman_rho_mean", "mean"),
    ).sort_values(["split", "miss_rate"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for split in sorted(g["split"].unique()):
        sub = g[g["split"] == split].sort_values("miss_rate")
        ax.plot(sub["miss_rate"], sub["tau"], marker="o", label=f"{split} Kendall τ")
        ax.plot(sub["miss_rate"], sub["rho"], marker="s", linestyle="--", label=f"{split} Spearman ρ")

    ax.set_xlabel("MCAR Missingness Rate")
    ax.set_ylabel("Ranking Correlation")
    ax.set_xlim(0, float(np.nanmax(g["miss_rate"])) if len(g) else 0.7)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    _save(fig, out_dir, "Fig2_ranking_stability_missingness.png")

# ============================================================
# Figure 3: Margin distribution + margin vs severity (Missingness)
# ============================================================
def fig_missingness_margin_distribution(margins: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(margins, ["margin_auc"], "phase5_margins_missingness.csv")
    margins = margins.copy()
    margins = _ensure_numeric(margins, ["margin_auc"])
    x = margins["margin_auc"].dropna().values

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(x, bins=40)
    ax.set_xlabel("AUROC Margin (Rank1 − Rank2)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    _save(fig, out_dir, "Fig3_margin_auc_distribution_missingness.png")

def fig_missingness_margin_vs_severity(margins: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(margins, ["split", "miss_rate", "seed", "margin_auc", "margin_ap", "margin_brier"],
                    "phase5_margins_missingness.csv")
    margins = margins.copy()
    margins = _ensure_numeric(margins, ["miss_rate", "margin_auc", "margin_ap", "margin_brier"])

    def _plot_one(metric: str, fname: str, ylab: str):
        g = margins.groupby(["split", "miss_rate"], as_index=False).agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
        ).sort_values(["split", "miss_rate"])
        g["std"] = g["std"].fillna(0.0)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        for split in sorted(g["split"].unique()):
            sub = g[g["split"] == split].sort_values("miss_rate")
            ax.plot(sub["miss_rate"], sub["mean"], marker="o", label=f"{split} mean")
            ax.fill_between(sub["miss_rate"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.2)

        ax.set_xlabel("MCAR Missingness Rate")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, float(np.nanmax(g["miss_rate"])) if len(g) else 0.7)
        ax.grid(True)
        ax.legend()
        _save(fig, out_dir, fname)

    _plot_one("margin_auc",   "Fig4_margin_auc_vs_missingness.png",   "AUROC Margin (Rank1 − Rank2)")
    _plot_one("margin_ap",    "Fig5_margin_ap_vs_missingness.png",    "AP Margin (Rank1 − Rank2)")
    _plot_one("margin_brier", "Fig6_margin_brier_vs_missingness.png",
              "Brier Margin (Rank2 − Rank1)  (positive ⇒ Rank1 better)")

# ============================================================
# Figure 7: Envelope sensitivity heatmap (Missingness)
# (Adjusted to match input columns flip_thr/tau_thr + stable_det/stable_tie)
# ============================================================
def fig_missingness_envelope_sensitivity_heatmap(env: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(env, ["split", "miss_rate", "flip_thr", "tau_thr", "stable_det", "stable_tie"],
                    "phase5_envelope_sensitivity_missingness.csv")

    df = env.copy()
    df = _ensure_numeric(df, ["miss_rate", "flip_thr", "tau_thr"])
    # booleans -> int for averaging
    df["stable_det"] = df["stable_det"].astype(int)
    df["stable_tie"] = df["stable_tie"].astype(int)

    def _plot(split: str, stable_col: str, suffix: str):
        sub = df[df["split"] == split].copy()
        piv = sub.groupby(["tau_thr", "flip_thr"], as_index=False)[stable_col].mean()
        taus = sorted(piv["tau_thr"].unique())
        flips = sorted(piv["flip_thr"].unique())

        mat = np.full((len(taus), len(flips)), np.nan, dtype=float)
        for i, t in enumerate(taus):
            for j, f in enumerate(flips):
                v = piv[(piv["tau_thr"] == t) & (piv["flip_thr"] == f)][stable_col]
                if len(v):
                    mat[i, j] = float(v.iloc[0])

        fig, ax = plt.subplots(figsize=FIGSIZE)
        im = ax.imshow(mat, aspect="auto", origin="lower")
        ax.set_xlabel("Flip Threshold (%)")
        ax.set_ylabel("Kendall τ Threshold")

        ax.set_xticks(np.arange(len(flips)))
        ax.set_xticklabels([f"{x:g}" for x in flips])
        ax.set_yticks(np.arange(len(taus)))
        ax.set_yticklabels([f"{x:g}" for x in taus])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean Stability (across severities)")

        title_mode = "Deterministic" if stable_col == "stable_det" else "Tie-aware"
        ax.set_title(f"Envelope Sensitivity (Missingness) — {split} — {title_mode}")
        _save(fig, out_dir, f"Fig7_envelope_sensitivity_heatmap_missingness_{split}_{suffix}.png")

    for split in sorted(df["split"].unique()):
        _plot(split, "stable_det", "det")
        _plot(split, "stable_tie", "tie")

# ============================================================
# Prevalence shift: Flip curve + Stability curve
# ============================================================
def fig_prevalence_flip_curve(env: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(env, ["split", "target_prev", "det_flip_pct", "tie_flip_pct"],
                    "phase5_envelope_sensitivity_prevalence_shift.csv")
    env = env.copy()
    env = _ensure_numeric(env, ["target_prev", "det_flip_pct", "tie_flip_pct"])

    g = env.groupby(["split", "target_prev"], as_index=False).agg(
        det_flip=("det_flip_pct", "mean"),
        tie_flip=("tie_flip_pct", "mean"),
    ).sort_values(["split", "target_prev"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for split in sorted(g["split"].unique()):
        sub = g[g["split"] == split].sort_values("target_prev")
        ax.plot(sub["target_prev"], sub["det_flip"], marker="o", label=f"{split} (deterministic)")
        ax.plot(sub["target_prev"], sub["tie_flip"], marker="s", linestyle="--", label=f"{split} (tie-aware)")

    ax.set_xlabel("Target Training Prevalence (native = -1)")
    ax.set_ylabel("Winner Flip (%)")
    ax.grid(True)
    ax.legend()
    _save(fig, out_dir, "Fig8_flip_curve_prevalence_shift.png")

def fig_prevalence_stability_curve(env: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(env, ["split", "target_prev", "kendall_tau_mean", "spearman_rho_mean"],
                    "phase5_envelope_sensitivity_prevalence_shift.csv")
    env = env.copy()
    env = _ensure_numeric(env, ["target_prev", "kendall_tau_mean", "spearman_rho_mean"])

    g = env.groupby(["split", "target_prev"], as_index=False).agg(
        tau=("kendall_tau_mean", "mean"),
        rho=("spearman_rho_mean", "mean"),
    ).sort_values(["split", "target_prev"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for split in sorted(g["split"].unique()):
        sub = g[g["split"] == split].sort_values("target_prev")
        ax.plot(sub["target_prev"], sub["tau"], marker="o", label=f"{split} Kendall τ")
        ax.plot(sub["target_prev"], sub["rho"], marker="s", linestyle="--", label=f"{split} Spearman ρ")

    ax.set_xlabel("Target Training Prevalence (native = -1)")
    ax.set_ylabel("Ranking Correlation")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    _save(fig, out_dir, "Fig9_ranking_stability_prevalence_shift.png")

# ============================================================
# Prevalence shift: Margin distribution + margin vs shift
# ============================================================
def fig_prevalence_margin_distribution(margins: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(margins, ["margin_auc"], "phase5_margins_prevalence_shift.csv")
    margins = margins.copy()
    margins = _ensure_numeric(margins, ["margin_auc"])
    x = margins["margin_auc"].dropna().values

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(x, bins=40)
    ax.set_xlabel("AUROC Margin (Rank1 − Rank2)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    _save(fig, out_dir, "Fig10_margin_auc_distribution_prevalence_shift.png")

def fig_prevalence_margin_vs_shift(margins: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(margins, ["split", "target_prev", "seed", "margin_auc", "margin_ap", "margin_brier"],
                    "phase5_margins_prevalence_shift.csv")
    margins = margins.copy()
    margins = _ensure_numeric(margins, ["target_prev", "margin_auc", "margin_ap", "margin_brier"])

    def _plot_one(metric: str, fname: str, ylab: str):
        g = margins.groupby(["split", "target_prev"], as_index=False).agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
        ).sort_values(["split", "target_prev"])
        g["std"] = g["std"].fillna(0.0)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        for split in sorted(g["split"].unique()):
            sub = g[g["split"] == split].sort_values("target_prev")
            ax.plot(sub["target_prev"], sub["mean"], marker="o", label=f"{split} mean")
            ax.fill_between(sub["target_prev"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.2)

        ax.set_xlabel("Target Training Prevalence (native = -1)")
        ax.set_ylabel(ylab)
        ax.grid(True)
        ax.legend()
        _save(fig, out_dir, fname)

    _plot_one("margin_auc",   "Fig11_margin_auc_vs_prevalence_shift.png",   "AUROC Margin (Rank1 − Rank2)")
    _plot_one("margin_ap",    "Fig12_margin_ap_vs_prevalence_shift.png",    "AP Margin (Rank1 − Rank2)")
    _plot_one("margin_brier", "Fig13_margin_brier_vs_prevalence_shift.png",
              "Brier Margin (Rank2 − Rank1)  (positive ⇒ Rank1 better)")

# ============================================================
# Prevalence shift: Envelope sensitivity heatmap
# (Adjusted to match input columns flip_thr/tau_thr + stable_det/stable_tie)
# ============================================================
def fig_prevalence_envelope_sensitivity_heatmap(env: pd.DataFrame, out_dir: str):
    set_plot_style()
    _assert_columns(env, ["split", "target_prev", "flip_thr", "tau_thr", "stable_det", "stable_tie"],
                    "phase5_envelope_sensitivity_prevalence_shift.csv")

    df = env.copy()
    df = _ensure_numeric(df, ["target_prev", "flip_thr", "tau_thr"])
    df["stable_det"] = df["stable_det"].astype(int)
    df["stable_tie"] = df["stable_tie"].astype(int)

    def _plot(split: str, stable_col: str, suffix: str):
        sub = df[df["split"] == split].copy()
        piv = sub.groupby(["tau_thr", "flip_thr"], as_index=False)[stable_col].mean()
        taus = sorted(piv["tau_thr"].unique())
        flips = sorted(piv["flip_thr"].unique())

        mat = np.full((len(taus), len(flips)), np.nan, dtype=float)
        for i, t in enumerate(taus):
            for j, f in enumerate(flips):
                v = piv[(piv["tau_thr"] == t) & (piv["flip_thr"] == f)][stable_col]
                if len(v):
                    mat[i, j] = float(v.iloc[0])

        fig, ax = plt.subplots(figsize=FIGSIZE)
        im = ax.imshow(mat, aspect="auto", origin="lower")
        ax.set_xlabel("Flip Threshold (%)")
        ax.set_ylabel("Kendall τ Threshold")

        ax.set_xticks(np.arange(len(flips)))
        ax.set_xticklabels([f"{x:g}" for x in flips])
        ax.set_yticks(np.arange(len(taus)))
        ax.set_yticklabels([f"{x:g}" for x in taus])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean Stability (across shift levels)")

        title_mode = "Deterministic" if stable_col == "stable_det" else "Tie-aware"
        ax.set_title(f"Envelope Sensitivity (Prevalence Shift) — {split} — {title_mode}")
        _save(fig, out_dir, f"Fig14_envelope_sensitivity_heatmap_prevalence_{split}_{suffix}.png")

    for split in sorted(df["split"].unique()):
        _plot(split, "stable_det", "det")
        _plot(split, "stable_tie", "tie")

# ============================================================
# Extra: Winner identity stacked rate (Missingness + Prevalence)
# ============================================================
def fig_winner_identity_rates(winners_missing: pd.DataFrame, winners_prev: pd.DataFrame, out_dir: str):
    set_plot_style()

    def _prep(df: pd.DataFrame, xcol: str, tag: str):
        _assert_columns(df, ["split", xcol, "winner_det"], f"phase5_winners_tieaware_{tag}.csv")
        df = df.copy()
        df = _ensure_numeric(df, [xcol])
        return df

    wM = _prep(winners_missing, "miss_rate", "missingness")
    wP = _prep(winners_prev, "target_prev", "prevalence_shift")

    # ---- Missingness
    for split in sorted(wM["split"].unique()):
        sub = wM[wM["split"] == split].copy()

        frac = sub.groupby(["miss_rate", "winner_det"]).size().reset_index(name="n")
        total = sub.groupby("miss_rate").size().reset_index(name="tot")
        frac = frac.merge(total, on="miss_rate", how="left")
        frac["pct"] = frac["n"] / frac["tot"]
        piv = frac.pivot(index="miss_rate", columns="winner_det", values="pct").fillna(0.0).sort_index()

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.stackplot(piv.index.values, [piv[c].values for c in piv.columns], labels=list(piv.columns))
        ax.set_xlabel("MCAR Missingness Rate")
        ax.set_ylabel("Winner Fraction Across Seeds")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.set_title(f"Winner Identity Rates (Deterministic) — Missingness — {split}")
        _save(fig, out_dir, f"Fig15_winner_identity_rates_missingness_{split}.png")

    # ---- Prevalence shift
    for split in sorted(wP["split"].unique()):
        sub = wP[wP["split"] == split].copy()

        frac = sub.groupby(["target_prev", "winner_det"]).size().reset_index(name="n")
        total = sub.groupby("target_prev").size().reset_index(name="tot")
        frac = frac.merge(total, on="target_prev", how="left")
        frac["pct"] = frac["n"] / frac["tot"]
        piv = frac.pivot(index="target_prev", columns="winner_det", values="pct").fillna(0.0).sort_index()

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.stackplot(piv.index.values, [piv[c].values for c in piv.columns], labels=list(piv.columns))
        ax.set_xlabel("Target Training Prevalence (native = -1)")
        ax.set_ylabel("Winner Fraction Across Seeds")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.set_title(f"Winner Identity Rates (Deterministic) — Prevalence Shift — {split}")
        _save(fig, out_dir, f"Fig16_winner_identity_rates_prevalence_{split}.png")

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase5_dir", type=str, default="/mnt/data",
                    help="Directory containing Phase 5 CSV outputs (default matches attached files).")
    ap.add_argument("--out_dir", type=str, default="PHASE5_FIGURES",
                    help="Directory to save figures.")
    args = ap.parse_args()

    d = args.phase5_dir
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    env_miss = _safe_read_csv(os.path.join(d, "phase5_envelope_sensitivity_missingness.csv"))
    env_prev = _safe_read_csv(os.path.join(d, "phase5_envelope_sensitivity_prevalence_shift.csv"))

    marg_miss = _safe_read_csv(os.path.join(d, "phase5_margins_missingness.csv"))
    marg_prev = _safe_read_csv(os.path.join(d, "phase5_margins_prevalence_shift.csv"))

    win_miss = _safe_read_csv(os.path.join(d, "phase5_winners_tieaware_missingness.csv"))
    win_prev = _safe_read_csv(os.path.join(d, "phase5_winners_tieaware_prevalence_shift.csv"))

    # Missingness
    fig_missingness_flip_curve(env_miss, out)
    fig_missingness_stability_curve(env_miss, out)
    fig_missingness_margin_distribution(marg_miss, out)
    fig_missingness_margin_vs_severity(marg_miss, out)
    fig_missingness_envelope_sensitivity_heatmap(env_miss, out)

    # Prevalence shift
    fig_prevalence_flip_curve(env_prev, out)
    fig_prevalence_stability_curve(env_prev, out)
    fig_prevalence_margin_distribution(marg_prev, out)
    fig_prevalence_margin_vs_shift(marg_prev, out)
    fig_prevalence_envelope_sensitivity_heatmap(env_prev, out)

    # Winner identity (det)
    fig_winner_identity_rates(win_miss, win_prev, out)

    print("DONE. Figures saved to:", os.path.abspath(out))

if __name__ == "__main__":
    main()