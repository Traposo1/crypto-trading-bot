@echo off
echo =============================================================
echo DIRECT EXECUTABLE BUILD SCRIPT (WINDOWS)
echo =============================================================

REM Ensure PyInstaller is installed
pip install pyinstaller>=5.1

REM Install required dependencies
pip install flask-socketio python-socketio python-engineio eventlet websocket-client dnspython

REM Run a direct PyInstaller command that works in most environments
pyinstaller --name=TradingBot --onefile --windowed ^
  --add-data=templates;templates ^
  --add-data=static;static ^
  --hidden-import=eventlet.hubs ^
  --hidden-import=eventlet.greenpools ^
  --hidden-import=dns ^
  --hidden-import=dns.resolver ^
  --hidden-import=socketio ^
  --hidden-import=flask_socketio ^
  --hidden-import=websocket ^
  executable_app.py

REM Create model directory if it doesn't exist
if not exist dist\model mkdir dist\model

REM Create data directory if it doesn't exist
if not exist dist\data mkdir dist\data

REM Copy README if it exists
if exist README_EXECUTABLE.md copy README_EXECUTABLE.md dist\README.md

echo =============================================================
echo Build completed! Executable is in the dist folder.
echo =============================================================

pause