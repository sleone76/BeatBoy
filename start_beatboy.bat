@echo off
TITLE BeatBoy Launcher
CLS

ECHO ===================================================
ECHO   BeatBoy - AI Music Generator (Fixed)
ECHO ===================================================
ECHO.

:: 1. Check for Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [FEHLER] Python wurde nicht gefunden!
    PAUSE
    EXIT /B
)

:: 2. Install dependencies (Quick check)
:: We use pip install just to be sure, it's fast if satisfied
ECHO [INFO] Pruefe Installation...
pip install fastapi uvicorn numpy scipy python-multipart >nul 2>&1

:: 3. Start Server
:: Important: We CD into backend so imports work correctly
:: And we use python -m uvicorn which uses the installed module
ECHO [INFO] Starte Backend Server...
cd backend
start "" cmd /c "python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000"
cd ..

:: Wait a bit for server to boot
timeout /t 4 /nobreak >nul

:: 4. Open Frontend
ECHO [INFO] Oeffne Web-Interface...
start frontend\index.html

ECHO.
ECHO [SUCCESS] BeatBoy sollte jetzt laufen!
ECHO Falls "Failed to fetch" kommt: Pruefe ob das schwarze Fenster noch offen ist.
PAUSE
