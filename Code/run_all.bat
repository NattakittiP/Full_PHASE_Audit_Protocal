@echo off
setlocal enabledelayedexpansion

set SKIP_PHASE4=0
set SKIP_FIGURES=0
set WITH_EXTERNAL=0
set RESUME_FROM=1

:parse
if "%~1"=="" goto done_parse
if "%~1"=="--skip-phase4"   ( set SKIP_PHASE4=1   & shift & goto parse )
if "%~1"=="--skip-figures"  ( set SKIP_FIGURES=1  & shift & goto parse )
if "%~1"=="--with-external" ( set WITH_EXTERNAL=1 & shift & goto parse )
if "%~1"=="--resume-from"   ( set RESUME_FROM=%~2 & shift & shift & goto parse )
shift & goto parse
:done_parse

set STEP=0
set START_TIME=%TIME%

echo.
echo ==================================================
echo   Protocol Audit Pipeline - IEEE JBHI Submission
echo ==================================================
echo   Started   : %DATE% %TIME%
echo   Directory : %CD%
echo ==================================================
echo.

echo [Pre-flight] Checking...
python --version
if errorlevel 1 ( echo [ERR] Python not found. & pause & exit /b 1 )

if not exist "phase1_3_main_audit_runner.py" (
    echo [ERR] Must run from the Code\ directory.
    pause & exit /b 1
)

python -c "import numpy,pandas,sklearn,scipy,tqdm,xgboost,matplotlib,openpyxl; print('  Packages : OK')"
if errorlevel 1 (
    echo [WARN] Some packages missing. Install with:
    echo        python -m pip install numpy pandas scikit-learn scipy tqdm xgboost matplotlib openpyxl
)

python -c "import os; missing=[f for f in ['full_analytic_dataset_mortality_all_admissions.csv','Synthetic_Dataset_1500_Patients_precise.csv'] if not os.path.exists(f)]; [print('  [MISSING]',f) for f in missing]; exit(len(missing))"
if errorlevel 1 (
    echo [ERR] Required data files missing.
    pause & exit /b 1
)
echo   Data files: OK
echo.

goto pipeline

:step
set /a STEP+=1
if !STEP! LSS %RESUME_FROM% (
    echo   [Step !STEP!] %~1 - skipped
    exit /b 0
)
echo.
echo --------------------------------------------------
echo   [Step !STEP!] %~1
echo   Time   : %TIME%
echo   Script : %~2
echo --------------------------------------------------
python %~2
if errorlevel 1 (
    echo.
    echo [FAILED] Step !STEP!: %~1
    echo          Retry: run_all.bat --resume-from !STEP!
    pause
    exit /b 1
)
echo   [OK] %~1
exit /b 0

:pipeline

call :step "Phase 1-3: Main Audit" phase1_3_main_audit_runner.py
if errorlevel 1 goto end

if "%SKIP_PHASE4%"=="0" (

    call :step "Phase 4a: Missingness Severity Sweep" phase4a_missingness_severity_sweep.py
    if errorlevel 1 goto end

    call :step "Phase 4b: Prevalence Shift Sweep" phase4b_prevalence_shift_sweep.py
    if errorlevel 1 goto end

    call :step "Phase 5: Tie-aware Ranking and Envelope" phase5_tieaware_envelope_margin_analysis.py
    if errorlevel 1 goto end

    if "%SKIP_FIGURES%"=="0" (
        set /a STEP+=1
        if !STEP! GEQ %RESUME_FROM% (
            echo.
            echo --------------------------------------------------
            echo   [Step !STEP!] Phase 5: Figure Generation
            echo --------------------------------------------------
            python phase5_figure_generator.py --phase5_dir PHASE5_ANALYSIS --out_dir PHASE5_FIGURES
            if errorlevel 1 ( echo [FAILED] Figures & pause & goto end )
            echo   [OK] Phase 5: Figure Generation
        )
    )

) else (
    echo   [Steps 2-5] Skipped
)

if "%WITH_EXTERNAL%"=="1" (
    call :step "External Cohort: eICU Runner" external_cohort_eicu_leakguard_runner.py
    if errorlevel 1 goto end
)

call :step "Final Summary Report" final_summary_report.py
if errorlevel 1 goto end

echo.
echo ==================================================
echo   PIPELINE COMPLETE
echo   Started : %START_TIME%
echo   Ended   : %TIME%
echo ==================================================
echo.
echo   final_summary\   - Master report (MD + XLSX + JSON)
echo.

:end
endlocal
