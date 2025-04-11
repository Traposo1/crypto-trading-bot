@echo off
echo ====================================================================
echo ADVANCED FLASK-SOCKETIO PATCH AND BUILD UTILITY FOR PYINSTALLER
echo ====================================================================

echo Installing required packages...
pip install pyinstaller flask-socketio eventlet dnspython

echo Creating essential directories...
if not exist model mkdir model
if not exist data mkdir data

echo Patching executable_app.py with socket patch...
python -c "import shutil; shutil.copy('executable_app.py', 'executable_app.py.bak')"

echo import no_stdin_fix > temp_file.txt
echo import flask_socketio_patch >> temp_file.txt
type executable_app.py >> temp_file.txt
move /y temp_file.txt executable_app.py
echo [√] Added patch imports to executable_app.py

echo Building executable with patched configuration...
pyinstaller --name=TradingBot ^
  --onefile ^
  --windowed ^
  --clean ^
  --noconfirm ^
  --hidden-import=flask_socketio_patch ^
  --hidden-import=no_stdin_fix ^
  --hidden-import=engineio.async_drivers.threading ^
  --hidden-import=eventlet.hubs ^
  --hidden-import=dns ^
  --hidden-import=dns.resolver ^
  --add-data=templates;templates ^
  --add-data=static;static ^
  --add-data=flask_socketio_patch.py;. ^
  --add-data=no_stdin_fix.py;. ^
  executable_app.py

echo Creating essential directories in distribution...
if not exist dist\model mkdir dist\model
if not exist dist\data mkdir dist\data

echo Copying README if available...
if exist README_EXECUTABLE.md copy README_EXECUTABLE.md dist\README.md

echo ====================================================================
echo BUILD PROCESS COMPLETED!
echo ====================================================================
echo.
echo Your executable is now available in the dist directory.
echo If any issues persist, try running the executable as administrator.
echo.
pause