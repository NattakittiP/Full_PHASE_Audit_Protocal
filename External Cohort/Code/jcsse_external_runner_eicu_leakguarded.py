# ============================================================
# jcsse_external_runner_eicu_leakguarded.py — External Cohort Runner (eICU)
# (Leakage-Guarded Edition) — "กัน Data Leakage แบบขั้นสุด"
#
# Locked to the SAME protocol semantics as jcsse_audit_runner_tqdm_hardened.py
#
# Adds STRICT leakage defenses at load time (feature hygiene only; no label-peeking):
#   (A) Drop known post-outcome / discharge-derived columns (name-based, deterministic)
#   (B) Drop obvious identifiers (ID-like, high-cardinality near-unique columns)
#   (C) HARD BLOCK: prevent "label-as-feature leakage" by dropping ALL columns
#       that look like label siblings (e.g., label24h/label48h/label_24h/label48hours)
#       except the chosen label_used.
#   (D) Optionally keep ONLY early-available features via prefix/regex rules (off by default)
#   (E) Save a leakage_guard_report.csv in results_dir (what was dropped & why)
#   (F) PRE-FLIGHT mode: run guard only (no training) to avoid wasting time
#
# NOTE:
# - We DO NOT use y to decide which features to drop (no label peeking).
# - Preflight checks are name-based only.
# ============================================================

import os
import json
import argparse
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple

import re
import numpy as np
import pandas as pd

# tqdm (same fallback style)
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    def tqdm(iterable=None, total=None, desc=None, leave=True, position=0, dynamic_ncols=True, **kwargs):
        if iterable is None:
            return range(total or 0)
        return iterable
    tqdm.write = print  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# Import the MAIN runner to guarantee identical semantics
# ------------------------------------------------------------
# Assumes this file is placed next to jcsse_audit_runner_tqdm_hardened.py
import jcsse_audit_runner_tqdm_hardened as core

# ------------------------------------------------------------
# Defaults (locked to match the main runner where applicable)
# ------------------------------------------------------------
MODELS = core.MODELS
PROTOCOLS = core.PROTOCOLS
SPLITS = core.SPLITS
SEEDS_20 = core.SEEDS_20

# Phase seeds (external)
SEED_PHASE1_EXT = 3036


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Leakage guard (deterministic, no label peeking)
# ------------------------------------------------------------
def _lower(s: str) -> str:
    return str(s).strip().lower()


def _pick_first_existing(cols: List[str], df_cols: List[str]) -> Optional[str]:
    s = set(df_cols)
    for c in cols:
        if c in s:
            return c
    return None


def _compile_patterns(pats: List[str]):
    return [re.compile(p, flags=re.IGNORECASE) for p in pats]


# (1) Strong post-outcome / discharge leakage patterns
# Keep broad (clinical tables frequently encode future/outcome info).
POST_OUTCOME_PATTERNS = _compile_patterns([
    # discharge / end-of-stay / timing
    r"discharge",
    r"endofstay", r"end_of_stay",
    r"unitdischargeoffset", r"hospitaldischargeoffset", r"dischargeoffset",
    r"dischargeyear", r"dischargemonth", r"dischargedate", r"dischargetime",

    # explicit outcome encoding
    r"expired", r"death", r"deceased", r"hospice",
    r"outcome",

    # mortality fields (if not the chosen label)
    r"mortality",

    # LOS often leaks post-outcome/time
    r"lengthofstay",
    r"\blos\b",
    r"stayduration",

    # readmit sometimes future-derived (depends on definition)
    r"readmit",

    # common ICU score/prediction leakage (safe even if absent)
    r"apache", r"aps", r"saps", r"sofa",
    r"predmort", r"pred_mort", r"predicted.*mort",
    r"patientresult",
])

# (2) Label-like sibling patterns (robust)
# We will also do an even stronger rule: ANY col that starts with "label" is a sibling label.
LABEL_SIBLING_PATTERNS = _compile_patterns([
    r"^label\s*\d+\s*h$",         # label24h, label48h (case/space tolerant)
    r"label[_\-]?\d+\s*h",        # label_24h, label-48h
    r"label.*hour",               # label48hours
])

# (3) ID-like columns to drop (except chosen group_col)
ID_LIKE_EXACT = set([
    "patientunitstayid", "uniquepid", "unitvisitnumber",
    "icustay_id", "stay_id", "subject_id", "hadm_id", "encounter_id",
    "visit_id", "admissionid", "admission_id",
    "case_id", "record_id", "row_id",
])

ID_LIKE_PATTERNS = _compile_patterns([
    r".*id$",              # endswith id
    r"^id$",
    r"patient.*id",
    r"encounter.*id",
    r"admission.*id",
])


def leakage_guard_drop_columns(
    df: pd.DataFrame,
    *,
    label_used: str,
    group_used: Optional[str],
    strict: bool,
    drop_extra_ids: bool,
    high_cardinality_thresh: float = 0.98,
    min_unique_for_hc: int = 200,
    keep_regex: Optional[str] = None,
    drop_regex: Optional[str] = None,
    extra_drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Returns: df_clean, report_rows

    - strict=True enables post-outcome patterns + high-cardinality drop
    - drop_extra_ids=True enables ID-like drop
    - extra_drop_cols: deterministic drop list (e.g., sibling labels)
    - keep_regex: if provided, keep ONLY columns matching regex (after mandatory keeps: label, group)
    - drop_regex: additionally drop columns matching regex
    """
    report: List[Dict[str, str]] = []
    cols = list(df.columns)

    must_keep = {label_used}
    if group_used:
        must_keep.add(group_used)

    to_drop = set()

    # A) Post-outcome / discharge leakage (name-based)
    if strict:
        for c in cols:
            if c in must_keep:
                continue
            lc = _lower(c)
            for pat in POST_OUTCOME_PATTERNS:
                if pat.search(lc):
                    to_drop.add(c)
                    report.append({"column": c, "reason": f"strict_post_outcome_pattern:{pat.pattern}"})
                    break

    # B) Obvious IDs (exact + pattern)
    if drop_extra_ids:
        for c in cols:
            if c in must_keep:
                continue
            lc = _lower(c)
            if lc in ID_LIKE_EXACT:
                to_drop.add(c)
                report.append({"column": c, "reason": "id_like_exact"})
                continue
            for pat in ID_LIKE_PATTERNS:
                if pat.search(lc):
                    to_drop.add(c)
                    report.append({"column": c, "reason": f"id_like_pattern:{pat.pattern}"})
                    break

    # C) High-cardinality near-unique (no label peeking; uses X only)
    if strict:
        n = len(df)
        if n > 0:
            for c in cols:
                if c in must_keep or c in to_drop:
                    continue
                nunique = df[c].nunique(dropna=True)
                if nunique < min_unique_for_hc:
                    continue
                ratio = nunique / max(n, 1)
                if ratio >= high_cardinality_thresh:
                    to_drop.add(c)
                    report.append({
                        "column": c,
                        "reason": f"high_cardinality_unique_ratio>={high_cardinality_thresh:.2f} (nunique={nunique}, n={n})"
                    })

    # D) Optional explicit drop_regex
    if drop_regex:
        rx = re.compile(drop_regex, flags=re.IGNORECASE)
        for c in cols:
            if c in must_keep:
                continue
            if rx.search(c):
                if c not in to_drop:
                    to_drop.add(c)
                    report.append({"column": c, "reason": f"user_drop_regex:{drop_regex}"})

    # E) Extra deterministic drop list (HARD BLOCK)
    if extra_drop_cols:
        for c in extra_drop_cols:
            if c in df.columns and c not in must_keep:
                if c not in to_drop:
                    to_drop.add(c)
                    report.append({"column": c, "reason": "hard_extra_drop_cols"})

    # Apply drop
    df2 = df.drop(columns=sorted(to_drop), errors="ignore").copy()

    # F) Optional keep_regex (whitelist) — applied AFTER dropping
    if keep_regex:
        rx = re.compile(keep_regex, flags=re.IGNORECASE)
        keep = [c for c in df2.columns if (c in must_keep) or rx.search(c)]
        drop2 = [c for c in df2.columns if c not in keep]
        if drop2:
            df2 = df2.drop(columns=drop2, errors="ignore").copy()
            for c in drop2:
                report.append({"column": c, "reason": f"user_keep_regex_filtered_out:{keep_regex}"})

    return df2, report


# ------------------------------------------------------------
# External dataset loader (eICU) — leakage guarded
# ------------------------------------------------------------
def load_dataset_eicu(
    path: str,
    label_col: Optional[str] = None,
    group_col: Optional[str] = None,
    drop_extra_ids: bool = True,
    strict_leakage_guard: bool = True,
    keep_regex: Optional[str] = None,
    drop_regex: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], List[str], List[str], str, str]:
    """
    Returns: X, y, groups, num_cols, cat_cols, label_col_used, group_col_used
    """
    df = core.normalize_columns(pd.read_csv(path))

    # label candidates (prefer simplest label if present)
    label_candidates = ["label", "label24h", "label48h", "mortality", "hospital_mortality", "icu_mortality"]
    label_used = label_col or _pick_first_existing(label_candidates, df.columns.tolist())
    if label_used is None or label_used not in df.columns:
        raise ValueError(f"[eICU] Could not find label column. Provide --label_col. Candidates tried: {label_candidates}")

    # group candidates (recommended hospital-level for S2)
    group_candidates = ["hospitalid", "uniquepid", "patientunitstayid", "wardid"]
    group_used = group_col or _pick_first_existing(group_candidates, df.columns.tolist())

    # --------------------------------------------------------
    # HARD BLOCK: prevent "label-as-feature" leakage (NO y usage)
    # Rule:
    #   drop ANY column that (a) startswith "label" OR matches label sibling patterns
    #   except label_used itself.
    # This will ALWAYS catch your dataset's label24h / label48h when label_used="label".
    # --------------------------------------------------------
    extra_drop_cols: List[str] = []
    for c in df.columns:
        if c == label_used:
            continue
        lc = _lower(c)

        # strongest rule: any column beginning with "label" is a sibling label
        if lc.startswith("label"):
            extra_drop_cols.append(c)
            continue

        # also keep regex patterns (covers weird naming)
        for pat in LABEL_SIBLING_PATTERNS:
            if pat.search(lc):
                extra_drop_cols.append(c)
                break

    extra_drop_cols = sorted(set(extra_drop_cols))

    # ---- Leakage guard (acts on full df, but DOES NOT use y) ----
    df_guarded, report = leakage_guard_drop_columns(
        df,
        label_used=label_used,
        group_used=group_used,
        strict=strict_leakage_guard,
        drop_extra_ids=drop_extra_ids,
        keep_regex=keep_regex,
        drop_regex=drop_regex,
        extra_drop_cols=extra_drop_cols,
    )

    # Save report if requested
    if report_path is not None:
        rep_df = pd.DataFrame(report)
        if len(rep_df) == 0:
            rep_df = pd.DataFrame([{"column": "", "reason": "NO_DROPS"}])
        rep_df = rep_df.sort_values(["reason", "column"]).reset_index(drop=True)
        rep_df.to_csv(report_path, index=False)

    # Re-check required cols still exist
    if label_used not in df_guarded.columns:
        raise ValueError(f"[eICU] label column '{label_used}' was dropped by guard. Fix keep/drop rules.")
    if group_used and group_used not in df_guarded.columns:
        # group is optional; proceed but S2 may be disabled later
        group_used = None

    # Extract y (binary)
    y = pd.to_numeric(df_guarded[label_used], errors="coerce").fillna(0).astype(int).values
    y = (y > 0).astype(int)

    groups = None
    if group_used is not None and group_used in df_guarded.columns:
        groups = df_guarded[group_used].values

    # Build X
    X = df_guarded.drop(columns=[label_used], errors="ignore").copy()

    # Determine numeric vs categorical (same heuristic family as core)
    cat_cols = []
    for c in X.columns:
        dt = str(X[c].dtype)
        if dt in ["object", "category", "bool", "string"] or dt.startswith("string"):
            cat_cols.append(c)
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Drop globally all-NaN columns (defensive)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols, errors="ignore")
        num_cols = [c for c in num_cols if c not in all_nan_cols]
        cat_cols = [c for c in cat_cols if c not in all_nan_cols]

    return X, y, groups, num_cols, cat_cols, label_used, (group_used or "")


# ------------------------------------------------------------
# Summaries / winners / flips (external)
# ------------------------------------------------------------
def summarize_configs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return core.summarize_configs(metrics_df)


def compute_winners(summary_df: pd.DataFrame) -> pd.DataFrame:
    return core.compute_winners(summary_df)


def compute_winner_flip_external(winners_df: pd.DataFrame, *, phase2_name: str, phase1_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Winner flip % across seeds for P0 only in PHASE2_REPRO_EXT,
    baseline = PHASE1_MAIN_EXT (seed SEED_PHASE1_EXT), per split.
    """
    base = {}
    for split_key in ["S1", "S2"]:
        row = winners_df[
            (winners_df["phase"] == phase1_name) &
            (winners_df["split"] == split_key) &
            (winners_df["protocol"] == "P0") &
            (winners_df["seed"] == SEED_PHASE1_EXT)
        ]
        if len(row) == 0:
            continue
        base[split_key] = row.iloc[0]["winner_model"]

    w2 = winners_df[
        (winners_df["phase"] == phase2_name) &
        (winners_df["protocol"] == "P0")
    ].copy()

    per_seed = w2[["split", "seed", "winner_model", "winner_auroc", "winner_ap", "winner_brier", "winner_ece"]].copy()
    per_seed = per_seed.sort_values(["split", "seed"]).reset_index(drop=True)

    return_df = []
    for split_key, sub in w2.groupby("split"):
        if split_key not in base:
            continue
        baseline = base[split_key]
        flips = (sub["winner_model"] != baseline).astype(int).values
        flip_pct = 100.0 * float(np.mean(flips)) if len(flips) else 0.0
        return_df.append({
            "dataset": "EICU",
            "protocol": "P0",
            "split": split_key,
            "baseline_seed": SEED_PHASE1_EXT,
            "baseline_winner": baseline,
            "n_seeds": int(len(flips)),
            "winner_flip_pct": float(flip_pct),
        })

    return pd.DataFrame(return_df), per_seed


# ------------------------------------------------------------
# Preflight: guard-only check (no training)
# ------------------------------------------------------------
def preflight_guard_only(
    *,
    data_path: str,
    label_col: Optional[str],
    group_col: Optional[str],
    results_dir: str,
    drop_extra_ids: bool,
    strict_guard: bool,
    keep_regex: Optional[str],
    drop_regex: Optional[str],
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    guard_report_path = os.path.join(results_dir, "leakage_guard_report.csv")

    tqdm.write(f"[{now_ts()}] PRE-FLIGHT: Loading eICU: {data_path}")
    X, _y, groups, num_cols, cat_cols, label_used, group_used = load_dataset_eicu(
        data_path,
        label_col=label_col,
        group_col=group_col,
        drop_extra_ids=drop_extra_ids,
        strict_leakage_guard=strict_guard,
        keep_regex=keep_regex,
        drop_regex=drop_regex,
        report_path=guard_report_path,
    )
    tqdm.write(f"[{now_ts()}] PRE-FLIGHT: label={label_used} | group={group_used if group_used else 'NONE'}")
    tqdm.write(f"[{now_ts()}] PRE-FLIGHT: guard report: {guard_report_path}")
    tqdm.write(f"[{now_ts()}] PRE-FLIGHT: X shape: {X.shape} | #num={len(num_cols)} | #cat={len(cat_cols)}")
    if groups is None:
        tqdm.write("[WARN] PRE-FLIGHT: group is None -> S2 will be skipped unless you provide --group_col hospitalid")

    # Name-based leftover risk scan (NO y usage)
    leftovers = []
    for c in X.columns:
        lc = _lower(c)
        if lc.startswith("label"):
            leftovers.append((c, "LEFTOVER_LABEL_LIKE"))
        else:
            for pat in POST_OUTCOME_PATTERNS:
                if pat.search(lc):
                    leftovers.append((c, f"LEFTOVER_POST_OUTCOME:{pat.pattern}"))
                    break

    if leftovers:
        tqdm.write("[FAIL] PRE-FLIGHT: Found suspicious leftover columns (name-based):")
        for c, why in leftovers[:50]:
            tqdm.write(f"  - {c}: {why}")
        tqdm.write("[FAIL] Fix guard rules before training. Exiting.")
        raise SystemExit(2)

    tqdm.write("[OK] PRE-FLIGHT: No suspicious (name-based) leftover columns. Safe to train.")
    raise SystemExit(0)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="eicu_2014_2015.csv", help="Path to eICU CSV.")
    ap.add_argument("--label_col", type=str, default="", help="Override label column (e.g., label, label24h).")
    ap.add_argument("--group_col", type=str, default="", help="Override group column for S2 (e.g., hospitalid).")
    ap.add_argument("--results_dir", type=str, default="results_eicu", help="Output folder.")
    ap.add_argument("--no_drop_extra_ids", action="store_true", help="Do not drop extra ID columns (not recommended).")

    # Leakage-guard controls
    ap.add_argument("--no_strict_leakage_guard", action="store_true",
                    help="Disable strict leakage guard (NOT recommended).")
    ap.add_argument("--keep_regex", type=str, default="",
                    help="Optional whitelist regex: keep ONLY columns matching this (plus label/group).")
    ap.add_argument("--drop_regex", type=str, default="",
                    help="Optional extra drop regex (applied in addition to strict rules).")

    # Preflight mode (new)
    ap.add_argument("--preflight_only", action="store_true",
                    help="Run leakage guard only (no training). Exits 0 if clean, exits 2 if suspicious columns remain.")
    args = ap.parse_args()

    label_col = args.label_col.strip() or None
    group_col = args.group_col.strip() or None
    results_dir = args.results_dir

    strict_guard = (not args.no_strict_leakage_guard)
    keep_regex = args.keep_regex.strip() or None
    drop_regex = args.drop_regex.strip() or None
    drop_extra_ids = (not args.no_drop_extra_ids)

    # PRE-FLIGHT: avoid wasting time
    if args.preflight_only:
        preflight_guard_only(
            data_path=args.data,
            label_col=label_col,
            group_col=group_col,
            results_dir=results_dir,
            drop_extra_ids=drop_extra_ids,
            strict_guard=strict_guard,
            keep_regex=keep_regex,
            drop_regex=drop_regex,
        )

    os.makedirs(results_dir, exist_ok=True)
    leak_dir = os.path.join(results_dir, "leakage_artifacts")
    os.makedirs(leak_dir, exist_ok=True)

    # IMPORTANT: redirect core to write leakage artifacts into this external folder
    core.RESULTS_DIR = results_dir
    core.LEAK_DIR = leak_dir

    config_log_path = os.path.join(results_dir, "config_log.jsonl")
    metrics_path = os.path.join(results_dir, "metrics_all.csv")
    summary_path = os.path.join(results_dir, "summary_by_config.csv")
    winners_path = os.path.join(results_dir, "winner_by_config.csv")
    flip_path = os.path.join(results_dir, "winner_flip_summary.csv")
    per_seed_path = os.path.join(results_dir, "winner_by_seed_phase2.csv")
    guard_report_path = os.path.join(results_dir, "leakage_guard_report.csv")

    if os.path.exists(config_log_path):
        os.remove(config_log_path)

    tqdm.write(f"[{now_ts()}] Loading eICU: {args.data}")
    X, y, groups, num_cols, cat_cols, label_used, group_used = load_dataset_eicu(
        args.data,
        label_col=label_col,
        group_col=group_col,
        drop_extra_ids=drop_extra_ids,
        strict_leakage_guard=strict_guard,
        keep_regex=keep_regex,
        drop_regex=drop_regex,
        report_path=guard_report_path,
    )
    tqdm.write(f"[{now_ts()}] eICU label: {label_used} | group(S2): {group_used if group_used else 'NONE'}")
    tqdm.write(f"[{now_ts()}] Leakage-guard report: {guard_report_path}")
    tqdm.write(f"[{now_ts()}] X shape after guard: {X.shape} | #num={len(num_cols)} | #cat={len(cat_cols)}")

    if groups is None:
        tqdm.write("[WARN] No group column found/selected. S2 (GroupKFold) will NOT be runnable. Provide --group_col hospitalid (recommended).")

    all_rows: List[Dict[str, Any]] = []

    # ---------------- PHASE 1 (External) ----------------
    phase1 = "PHASE1_MAIN_EXT"
    total_runs = 2 * 4 * len(MODELS)
    pbar = tqdm(total=total_runs, desc="PHASE1_MAIN_EXT (eICU matrix)", dynamic_ncols=True)
    for split_key in ["S1", "S2"]:
        if split_key == "S2" and groups is None:
            # skip S2 safely
            for _protocol in PROTOCOLS:
                for _model_key in MODELS:
                    pbar.update(1)
            continue

        for protocol in PROTOCOLS:
            for model_key in MODELS:
                cfg = {
                    "timestamp": now_ts(),
                    "phase": phase1,
                    "dataset": "EICU",
                    "split": split_key,
                    "protocol": protocol,
                    "model": model_key,
                    "seed": SEED_PHASE1_EXT,
                    "label_col": label_used,
                    "group_col": group_used,
                    "strict_leakage_guard": bool(strict_guard),
                    "keep_regex": keep_regex or "",
                    "drop_regex": drop_regex or "",
                }
                jsonl_append(config_log_path, cfg)

                store_oof = (protocol == "P0")
                rows, oof = core.run_config(
                    phase=phase1,
                    dataset_tag="EICU",
                    X=X,
                    y=y,
                    split_key=split_key,
                    protocol=protocol,
                    model_key=model_key,
                    seed=SEED_PHASE1_EXT,
                    groups=groups if split_key == "S2" else None,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    config_tag=phase1,
                    store_oof=store_oof,
                )
                all_rows.extend(rows)

                if store_oof and oof is not None:
                    npz_path = os.path.join(results_dir, f"oof_P0_{split_key}_{model_key}.npz")
                    np.savez_compressed(npz_path, **oof)

                pbar.update(1)
    pbar.close()

    # ---------------- PHASE 2 (External) ----------------
    phase2 = "PHASE2_REPRO_EXT"
    total_runs = 2 * len(SEEDS_20) * len(MODELS)
    pbar = tqdm(total=total_runs, desc="PHASE2_REPRO_EXT (eICU, P0, 20 seeds)", dynamic_ncols=True)
    for split_key in ["S1", "S2"]:
        if split_key == "S2" and groups is None:
            pbar.update(len(SEEDS_20) * len(MODELS))
            continue

        for seed in SEEDS_20:
            for model_key in MODELS:
                cfg = {
                    "timestamp": now_ts(),
                    "phase": phase2,
                    "dataset": "EICU",
                    "split": split_key,
                    "protocol": "P0",
                    "model": model_key,
                    "seed": seed,
                    "label_col": label_used,
                    "group_col": group_used,
                    "strict_leakage_guard": bool(strict_guard),
                    "keep_regex": keep_regex or "",
                    "drop_regex": drop_regex or "",
                }
                jsonl_append(config_log_path, cfg)

                rows, _ = core.run_config(
                    phase=phase2,
                    dataset_tag="EICU",
                    X=X,
                    y=y,
                    split_key=split_key,
                    protocol="P0",
                    model_key=model_key,
                    seed=seed,
                    groups=groups if split_key == "S2" else None,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    config_tag=phase2,
                    store_oof=False,
                )
                all_rows.extend(rows)
                pbar.update(1)
    pbar.close()

    # ---------------- Save outputs ----------------
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(metrics_path, index=False)

    summary_df = summarize_configs(metrics_df)
    summary_df.to_csv(summary_path, index=False)

    winners_df = compute_winners(summary_df)
    winners_df.to_csv(winners_path, index=False)

    flip_df, per_seed = compute_winner_flip_external(winners_df, phase2_name=phase2, phase1_name=phase1)
    flip_df.to_csv(flip_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)

    # Canonical OOF files for P0 S1/S2
    for split_key in ["S1", "S2"]:
        row = winners_df[
            (winners_df["phase"] == phase1) &
            (winners_df["split"] == split_key) &
            (winners_df["protocol"] == "P0") &
            (winners_df["seed"] == SEED_PHASE1_EXT)
        ]
        if len(row) == 0:
            continue
        winner_model = row.iloc[0]["winner_model"]
        src = os.path.join(results_dir, f"oof_P0_{split_key}_{winner_model}.npz")
        dst = os.path.join(results_dir, f"oof_P0_{split_key}.npz")
        if os.path.exists(src):
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())

    print("DONE (External eICU, Leakage-Guarded).")
    print(f"- results_dir: {results_dir}")
    print(f"- leakage guard report: {guard_report_path}")
    print(f"- metrics:     {metrics_path}")
    print(f"- summary:     {summary_path}")
    print(f"- winners:     {winners_path}")
    print(f"- flip:        {flip_path}")
    print(f"- per-seed:    {per_seed_path}")
    print(f"- oof canon:   {os.path.join(results_dir, 'oof_P0_S1.npz')} and {os.path.join(results_dir, 'oof_P0_S2.npz')}")
    print(f"- leakage dir: {leak_dir}")
    print(f"- config log:  {config_log_path}")


if __name__ == "__main__":
    main()