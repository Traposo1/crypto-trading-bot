@echo off
echo ==== Crypto Trading Bot Executable Builder ====
echo This script will build an executable version of the Trading Bot

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher and try again
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set python_version=%%i
echo Using Python version: %python_version%

REM Check if PyInstaller is installed
python -c "import PyInstaller" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if %ERRORLEVEL% neq 0 (
        echo Failed to install PyInstaller. Please install manually with:
        echo pip install pyinstaller
        pause
        exit /b 1
    )
)

REM Ensure model directory exists
if not exist "model" mkdir model

REM Create empty model files if they don't exist
for %%f in (model.pkl scaler.pkl feature_list.pkl) do (
    if not exist "model\%%f" (
        echo Creating empty model file: %%f
        type nul > "model\%%f"
    )
)

REM Run PyInstaller
echo Starting build process with PyInstaller...
echo This may take several minutes...

REM Windows specific PyInstaller command
pyinstaller --name=TradingBot --onefile --windowed ^
    --add-data="templates;templates" ^
    --add-data="static;static" ^
    --hidden-import=sklearn.ensemble ^
    --hidden-import=sklearn.tree ^
    --hidden-import=sklearn.preprocessing ^
    --hidden-import=sklearn.neural_network ^
    --hidden-import=flask_socketio ^
    --hidden-import=eventlet ^
    --hidden-import=eventlet.hubs ^
    --hidden-import=dns ^
    --hidden-import=dns.resolver ^
    --hidden-import=engineio.async_drivers ^
    --hidden-import=engineio.async_drivers.threading ^
    --hidden-import=socketio ^
    --hidden-import=websocket ^
    --hidden-import=websocket._socket ^
    --hidden-import=pandas_ta ^
    --hidden-import=flask ^
    --hidden-import=ccxt ^
    --hidden-import=webbrowser ^
    executable_app.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo Build completed successfully!
    echo The executable file is in the 'dist' folder.
    echo.
    echo Instructions for distribution:
    echo 1. Share the TradingBot.exe file from the 'dist' folder
    echo 2. Include the 'model' folder with the executable
    echo 3. Optionally include a config.json file for default settings
    echo.
) else (
    echo Build failed. Check the output for errors.
    pause
    exit /b 1
)

pause