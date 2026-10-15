@echo off
cd /d "%~dp0"
set "AV_ASD_RUNTIME_CACHE=%~dp0runtime_cache"
if not exist "%AV_ASD_RUNTIME_CACHE%\temp" mkdir "%AV_ASD_RUNTIME_CACHE%\temp"
if not exist "%AV_ASD_RUNTIME_CACHE%\numba" mkdir "%AV_ASD_RUNTIME_CACHE%\numba"
if not exist "%AV_ASD_RUNTIME_CACHE%\matplotlib" mkdir "%AV_ASD_RUNTIME_CACHE%\matplotlib"
set "TEMP=%AV_ASD_RUNTIME_CACHE%\temp"
set "TMP=%AV_ASD_RUNTIME_CACHE%\temp"
set "NUMBA_CACHE_DIR=%AV_ASD_RUNTIME_CACHE%\numba"
set "MPLCONFIGDIR=%AV_ASD_RUNTIME_CACHE%\matplotlib"
"%USERPROFILE%\anaconda3\envs\practice\python.exe" -m streamlit run "dashboard\research_app.py" --server.address 127.0.0.1 --server.port 8503
pause
