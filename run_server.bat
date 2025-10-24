@echo off
echo ===================================
echo Starting Deepfake Detection Server
echo ===================================
echo.
echo Server will run on: http://localhost:5001
echo Keep this window open!
echo Press Ctrl+C to stop
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app.py

pause
