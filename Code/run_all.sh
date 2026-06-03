#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full Protocol Audit Pipeline Runner
# IEEE JBHI Submission
#
# USAGE:
#   bash run_all.sh [OPTIONS]
#
# OPTIONS:
#   --with-external   Also run external cohort (eICU) runner
#   --skip-figures    Skip figure generation (saves time)
#   --skip-phase4     Skip Phase 4a + 4b (and Phase 5 which depends on them)
#   --resume-from N   Resume from step N (1–7); skips earlier steps
#   --dry-run         Print commands without executing
#
# REQUIREMENTS:
#   - Must be run from the Code/ directory
#   - Python 3.9+ with: numpy pandas scikit-learn scipy tqdm xgboost
#                        matplotlib openpyxl
#   - Git Bash / WSL / Linux / macOS
#
# EXAMPLES:
#   bash run_all.sh
#   bash run_all.sh --with-external
#   bash run_all.sh --skip-figures --resume-from 4
#   bash run_all.sh --dry-run
# =============================================================================

set -euo pipefail

# ─── ANSI Colors ─────────────────────────────────────────────────────────────
RED='\033[0;31m';   GREEN='\033[0;32m';  YELLOW='\033[1;33m'
BLUE='\033[0;34m';  CYAN='\033[0;36m';  MAGENTA='\033[0;35m'
BOLD='\033[1m';     DIM='\033[2m';      NC='\033[0m'

# ─── Flags ───────────────────────────────────────────────────────────────────
WITH_EXTERNAL=false
SKIP_FIGURES=false
SKIP_PHASE4=false
RESUME_FROM=1
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --with-external)      WITH_EXTERNAL=true ;;
        --skip-figures)       SKIP_FIGURES=true ;;
        --skip-phase4)        SKIP_PHASE4=true ;;
        --resume-from=*)      RESUME_FROM="${arg#*=}" ;;
        --dry-run)            DRY_RUN=true ;;
        -h|--help)
            sed -n '4,20p' "$0"; exit 0 ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"; exit 1 ;;
    esac
done

# ─── Timing state ────────────────────────────────────────────────────────────
PIPELINE_START=$(date +%s)
STEP_NAMES=()
STEP_ELAPSED=()
STEP_STATUS=()

format_time() {
    local s=$1
    if   [ "$s" -lt 60 ];   then printf "%ds" "$s"
    elif [ "$s" -lt 3600 ]; then printf "%dm %ds" "$((s/60))" "$((s%60))"
    else                          printf "%dh %dm %ds" "$((s/3600))" "$(((s%3600)/60))" "$((s%60))"
    fi
}

# ─── Step runner ─────────────────────────────────────────────────────────────
CURRENT_STEP=0

run_step() {
    local num="$1"          # "1/7"
    local name="$2"         # human label
    local script="$3"       # python script name
    shift 3
    local extra_args=("$@") # optional extra CLI args

    CURRENT_STEP=$((CURRENT_STEP + 1))

    # Resume check
    if [ "$CURRENT_STEP" -lt "$RESUME_FROM" ]; then
        echo -e "${DIM}  [Step ${num}] ${name}  —  skipped (--resume-from=${RESUME_FROM})${NC}"
        STEP_NAMES+=("$name")
        STEP_ELAPSED+=("—")
        STEP_STATUS+=("skipped")
        return 0
    fi

    echo ""
    echo -e "${BLUE}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│${NC}  ${BOLD}Step ${num}${NC}  ${CYAN}${name}${NC}"
    echo -e "${BLUE}│${NC}  ${DIM}$(date '+%Y-%m-%d %H:%M:%S')  •  ${script}${NC}"
    echo -e "${BLUE}└─────────────────────────────────────────────────────────────┘${NC}"

    local t_start t_end elapsed
    t_start=$(date +%s)

    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${YELLOW}[DRY-RUN]${NC} python ${script} ${extra_args[*]:-}"
        elapsed=0
        STEP_NAMES+=("$name")
        STEP_ELAPSED+=("dry-run")
        STEP_STATUS+=("dry-run")
        return 0
    fi

    if python "$script" "${extra_args[@]}"; then
        t_end=$(date +%s)
        elapsed=$((t_end - t_start))
        echo ""
        echo -e "  ${GREEN}✓  Completed in $(format_time $elapsed)${NC}"
        STEP_NAMES+=("$name")
        STEP_ELAPSED+=("$(format_time $elapsed)")
        STEP_STATUS+=("OK")
    else
        t_end=$(date +%s)
        elapsed=$((t_end - t_start))
        echo ""
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}  ✗  Step ${num} FAILED after $(format_time $elapsed)${NC}"
        echo -e "${RED}  Fix the error above, then re-run with: --resume-from=${CURRENT_STEP}${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        STEP_NAMES+=("$name")
        STEP_ELAPSED+=("$(format_time $elapsed)")
        STEP_STATUS+=("FAILED")
        exit 1
    fi
}

# ─── Print final summary table ────────────────────────────────────────────────
print_summary() {
    local total
    total=$(( $(date +%s) - PIPELINE_START ))

    echo ""
    echo -e "${BOLD}${GREEN}╔═════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${GREEN}║              PIPELINE COMPLETE                              ║${NC}"
    echo -e "${BOLD}${GREEN}╚═════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${BOLD}Step-by-Step Timing${NC}"
    echo -e "  ${DIM}──────────────────────────────────────────────────────────${NC}"

    for i in "${!STEP_NAMES[@]}"; do
        local st="${STEP_STATUS[$i]}"
        local el="${STEP_ELAPSED[$i]}"
        local nm="${STEP_NAMES[$i]}"
        if   [ "$st" = "OK"      ]; then echo -e "  ${GREEN}✓${NC}  ${nm}  ${DIM}→${NC}  ${el}"
        elif [ "$st" = "skipped" ]; then echo -e "  ${DIM}–  ${nm}  →  skipped${NC}"
        elif [ "$st" = "dry-run" ]; then echo -e "  ${YELLOW}○  ${nm}  →  dry-run${NC}"
        else                             echo -e "  ${RED}✗  ${nm}  →  ${el}  [FAILED]${NC}"
        fi
    done

    echo -e "  ${DIM}──────────────────────────────────────────────────────────${NC}"
    echo -e "  ${BOLD}Total elapsed : $(format_time $total)${NC}"
    echo ""
    echo -e "  ${BOLD}Output locations:${NC}"
    echo -e "  ${CYAN}  results/${NC}                     Phase 1-3 metrics + winners"
    if [ "$SKIP_PHASE4" = false ]; then
        echo -e "  ${CYAN}  PHASE4_SEVERITY_SWEEP/${NC}      Missingness sweep"
        echo -e "  ${CYAN}  PHASE4B_PREVALENCE_SHIFT/${NC}   Prevalence shift sweep"
        echo -e "  ${CYAN}  PHASE5_ANALYSIS/${NC}            Tie-aware + margins"
    fi
    if [ "$SKIP_FIGURES" = false ] && [ "$SKIP_PHASE4" = false ]; then
        echo -e "  ${CYAN}  PHASE5_FIGURES/${NC}             All paper figures (PNG)"
    fi
    if [ "$WITH_EXTERNAL" = true ]; then
        echo -e "  ${CYAN}  results_external/${NC}           External cohort (eICU)"
    fi
    echo -e "  ${GREEN}  final_summary/${NC}              ${BOLD}← Master report (MD + XLSX + JSON)${NC}"
    echo ""
}

# ─── Pre-flight check ─────────────────────────────────────────────────────────
preflight_check() {
    echo -e "${YELLOW}[Pre-flight] Checking environment...${NC}"

    # Python version
    local pyver
    pyver=$(python --version 2>&1)
    echo -e "  Python  : ${pyver}"

    # Working directory
    if [ ! -f "phase1_3_main_audit_runner.py" ]; then
        echo -e "${RED}  [ERR] Must be run from the Code/ directory.${NC}"
        echo -e "${RED}        Expected: phase1_3_main_audit_runner.py in current dir.${NC}"
        exit 1
    fi
    echo -e "  Dir     : $(pwd)  ${GREEN}✓${NC}"

    # Required packages
    python - <<'PYCHECK'
import sys
pkgs = {
    "numpy":        "numpy",
    "pandas":       "pandas",
    "sklearn":      "scikit-learn",
    "scipy":        "scipy",
    "tqdm":         "tqdm",
    "xgboost":      "xgboost",
    "matplotlib":   "matplotlib",
    "openpyxl":     "openpyxl",
}
missing = []
for mod, pip_name in pkgs.items():
    try:
        __import__(mod)
    except ImportError:
        missing.append(pip_name)
if missing:
    print(f"  [WARN] Missing packages: {', '.join(missing)}")
    print(f"         Install with:  pip install {' '.join(missing)}")
else:
    print("  Packages: all required packages found  ✓")
PYCHECK

    # Data files
    echo -e "  Checking data files..."
    local missing_data=false
    for f in \
        "full_analytic_dataset_mortality_all_admissions.csv" \
        "Synthetic_Dataset_1500_Patients_precise.csv"; do
        if [ ! -f "$f" ]; then
            echo -e "  ${RED}  [MISSING] $f${NC}"
            missing_data=true
        else
            echo -e "  ${GREEN}  [OK]     $f${NC}"
        fi
    done
    if [ "$missing_data" = true ]; then
        echo -e "${RED}[ERR] Required data files are missing. Pipeline cannot start.${NC}"
        exit 1
    fi

    echo ""
}

# ─── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔═════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   Protocol Audit Pipeline  •  IEEE JBHI Submission          ║${NC}"
echo -e "${BOLD}${CYAN}╚═════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Configuration:${NC}"
echo -e "  ${DIM}──────────────────────────────────────────────${NC}"
echo -e "  Started        : $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "  Directory      : $(pwd)"
echo -e "  External cohort: ${WITH_EXTERNAL}"
echo -e "  Generate figs  : $( [ "$SKIP_FIGURES" = true ] && echo 'no (--skip-figures)' || echo 'yes' )"
echo -e "  Phase 4 + 5    : $( [ "$SKIP_PHASE4"  = true ] && echo 'no (--skip-phase4)'  || echo 'yes' )"
echo -e "  Resume from    : step ${RESUME_FROM}"
echo -e "  Dry run        : ${DRY_RUN}"
echo -e "  ${DIM}──────────────────────────────────────────────${NC}"
echo ""

# ─── Pre-flight ───────────────────────────────────────────────────────────────
preflight_check

# ─── Pipeline ─────────────────────────────────────────────────────────────────

# Step 1 — Phase 1-3: Main audit
run_step "1/6" \
    "Phase 1-3  │ Main Audit (Leakage + Reproducibility + Synthetic)" \
    "phase1_3_main_audit_runner.py"

if [ "$SKIP_PHASE4" = false ]; then

    # Step 2 — Phase 4a: Missingness sweep
    run_step "2/6" \
        "Phase 4a   │ Missingness Severity Sweep  (MCAR 0–70%)" \
        "phase4a_missingness_severity_sweep.py"

    # Step 3 — Phase 4b: Prevalence shift
    run_step "3/6" \
        "Phase 4b   │ Prevalence / Label-Shift Sweep" \
        "phase4b_prevalence_shift_sweep.py"

    # Step 4 — Phase 5: Tie-aware + envelope + margins
    run_step "4/6" \
        "Phase 5    │ Tie-aware Ranking + Envelope Sensitivity + Margins" \
        "phase5_tieaware_envelope_margin_analysis.py"

    # Step 5a — Figure generation (optional)
    if [ "$SKIP_FIGURES" = false ]; then
        run_step "5a/6" \
            "Phase 5    │ Figure Generation (paper-quality PNG)" \
            "phase5_figure_generator.py" \
            "--phase5_dir" "PHASE5_ANALYSIS" \
            "--out_dir"    "PHASE5_FIGURES"
    else
        echo -e "${DIM}  [Step 5a] Figure generation — skipped (--skip-figures)${NC}"
        STEP_NAMES+=("Phase 5 | Figure Generation")
        STEP_ELAPSED+=("—")
        STEP_STATUS+=("skipped")
    fi

else
    echo -e "${DIM}  [Steps 2-5] Phase 4a / 4b / 5 — skipped (--skip-phase4)${NC}"
    for nm in \
        "Phase 4a  | Missingness Sweep" \
        "Phase 4b  | Prevalence Sweep" \
        "Phase 5   | Tie-aware Analysis" \
        "Phase 5   | Figure Generation"; do
        STEP_NAMES+=("$nm")
        STEP_ELAPSED+=("—")
        STEP_STATUS+=("skipped")
    done
fi

# Step 5b — External cohort (optional)
if [ "$WITH_EXTERNAL" = true ]; then
    run_step "5b/6" \
        "External   │ eICU Cohort — Leakage-Guarded Runner" \
        "external_cohort_eicu_leakguard_runner.py"
fi

# Step 6 — Final summary report
run_step "6/6" \
    "Summary    │ Final Report Generator (MD + XLSX + CSV + JSON)" \
    "final_summary_report.py"

# ─── Done ─────────────────────────────────────────────────────────────────────
print_summary
