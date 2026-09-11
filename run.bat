@echo off
setlocal

cd /d "%~dp0"

set "NO_PAUSE="
if /i "%~1"=="--no-pause" set "NO_PAUSE=1"

echo Creating virtual environment if needed...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo Running start.py...
python start.py

echo.
if not defined NO_PAUSE pause
