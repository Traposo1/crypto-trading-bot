@echo off
echo ====================================================================
echo SIMPLE CONSOLE EXECUTABLE BUILD UTILITY
echo ====================================================================

echo Installing required packages...
pip install pyinstaller flask-socketio eventlet dnspython

echo Creating essential directories...
if not exist model mkdir model
if not exist data mkdir data

echo Building console executable for easier debugging...
pyinstaller --name=TradingBot_Console ^
  --console ^
  --clean ^
  --noconfirm ^
  --hidden-import=flask_socketio ^
  --hidden-import=eventlet.hubs ^
  --hidden-import=dns ^
  --hidden-import=dns.resolver ^
  --hidden-import=engineio.async_drivers.threading ^
  --add-data=flask_socketio_patch.py;. ^
  --add-data=no_stdin_fix.py;. ^
  --add-data=templates;templates ^
  --add-data=static;static ^
  executable_app.py

echo Creating essential directories in distribution...
if not exist dist\model mkdir dist\model
if not exist dist\data mkdir dist\data

echo ====================================================================
echo BUILD PROCESS COMPLETED!
echo ====================================================================
echo.
echo Your console executable is now available in the dist directory.
echo This version will show any errors in a console window.
echo.
pause