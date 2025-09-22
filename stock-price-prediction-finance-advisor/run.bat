@echo off
REM Enhanced Stock Analyzer - Quick Setup & Launch (Windows)
REM Made by Neonite - Production Version

echo.
echo 📈 Enhanced Stock Analyzer
echo Made by Neonite
echo =========================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM Install dependencies
echo 📦 Installing dependencies...
venv\Scripts\pip.exe install --upgrade pip -q
venv\Scripts\pip.exe install -r requirements.txt -q

REM Launch GUI
echo ✅ Setup complete!
echo.
echo 🚀 Launching GUI with Interactive Charts...
venv\Scripts\python.exe gui.py

pause
