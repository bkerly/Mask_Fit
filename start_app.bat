@echo off
REM NIOSH Mask Fitting System - Windows Startup Script

echo ==========================================
echo NIOSH Respirator Mask Fitting System
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed.
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo Dependencies already installed.
)

echo.
echo ==========================================
echo Starting application...
echo ==========================================
echo.
echo The application will open in your default web browser.
echo If it doesn't open automatically, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Run the Streamlit app
streamlit run mask_fitting_app.py

pause
