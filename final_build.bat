@echo off
echo ====================================================================
echo FINAL BUILD SCRIPT WITH ALL FIXES
echo ====================================================================

echo Running eventlet fix script to handle missing modules...
python eventlet_fix.py

echo Installing required packages...
pip install pyinstaller flask-socketio eventlet dnspython

echo Creating essential directories...
if not exist model mkdir model
if not exist data mkdir data

echo Preparing executable_app.py with fixes...
python -c "import shutil; shutil.copy('executable_app.py', 'executable_app.py.bak')"

echo import no_stdin_fix > temp_file.txt
echo import flask_socketio_patch >> temp_file.txt
type executable_app.py >> temp_file.txt
move /y temp_file.txt executable_app.py
echo [OK] Added patch imports to executable_app.py

echo Building executable with all fixes applied...
pyinstaller --name=TradingBot ^
  --onefile ^
  --windowed ^
  --clean ^
  --noconfirm ^
  --hidden-import=flask_socketio_patch ^
  --hidden-import=no_stdin_fix ^
  --hidden-import=engineio.async_drivers.threading ^
  --hidden-import=eventlet.hubs ^
  --hidden-import=eventlet.hubs.epolls ^
  --hidden-import=dns ^
  --hidden-import=dns.resolver ^
  --add-data=flask_socketio_patch.py;. ^
  --add-data=no_stdin_fix.py;. ^
  --add-data=templates;templates ^
  --add-data=static;static ^
  executable_app.py

echo Creating essential directories in distribution...
if not exist dist\model mkdir dist\model
if not exist dist\data mkdir dist\data

echo ====================================================================
echo BUILDING CONSOLE VERSION FOR DEBUGGING
echo ====================================================================

echo Building console executable for easier debugging...
pyinstaller --name=TradingBot_Console ^
  --console ^
  --clean ^
  --noconfirm ^
  --hidden-import=flask_socketio_patch ^
  --hidden-import=no_stdin_fix ^
  --hidden-import=engineio.async_drivers.threading ^
  --hidden-import=eventlet.hubs ^
  --hidden-import=eventlet.hubs.epolls ^
  --hidden-import=dns ^
  --hidden-import=dns.resolver ^
  --add-data=flask_socketio_patch.py;. ^
  --add-data=no_stdin_fix.py;. ^
  --add-data=templates;templates ^
  --add-data=static;static ^
  executable_app.py

echo Restoring original executable_app.py...
python -c "import shutil; shutil.copy('executable_app.py.bak', 'executable_app.py')"

echo Copying README if available...
if exist executable_readme.md copy executable_readme.md dist\README.md

echo ====================================================================
echo BUILD PROCESS COMPLETED!
echo ====================================================================
echo.
echo Your executables are now available in the dist directory:
echo - TradingBot.exe (Windowed version)
echo - TradingBot_Console.exe (Console version for debugging)
echo.
echo If the windowed version doesn't work, try the console version
echo to see detailed error messages.
echo.
pause