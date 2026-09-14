@echo off
setlocal
cd /d "%~dp0"
if defined OCEANMIND_PYTHON (
  "%OCEANMIND_PYTHON%" "%~dp0server.py" %*
) else (
  python "%~dp0server.py" %*
)
exit /b %errorlevel%
