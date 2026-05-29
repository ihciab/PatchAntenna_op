@echo off
setlocal

if not defined WMIC_COMPAT_PYTHON (
    set "WMIC_COMPAT_PYTHON=python"
)

"%WMIC_COMPAT_PYTHON%" "%~dp0wmic_compat.py" %*
exit /b %ERRORLEVEL%
