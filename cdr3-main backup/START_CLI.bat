@echo off
cd /d %~dp0

REM Activate the virtual environment
call venv\Scripts\activate

REM Run the CLI
python cdr3_cli.py

REM Keep window open after exit
pause
