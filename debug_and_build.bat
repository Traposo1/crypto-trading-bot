@echo off
echo ====================================================================
echo DEBUG VERSION - CONSOLE EXECUTABLE BUILD SCRIPT
echo ====================================================================

echo Installing required packages...
pip install pyinstaller flask-socketio eventlet dnspython

echo Creating essential directories...
if not exist model mkdir model
if not exist data mkdir data

echo Patching executable_app.py with stdout redirection...
python -c "import shutil; shutil.copy('executable_app.py', 'executable_app.py.bak')"

echo import sys > temp_file.txt
echo import os >> temp_file.txt
echo. >> temp_file.txt
echo # Redirect all output to a log file and console >> temp_file.txt
echo log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug.log') >> temp_file.txt
echo sys.stdout = open(log_path, 'w') >> temp_file.txt
echo print("Debug log started at: {}".format(time.strftime("%%Y-%%m-%%d %%H:%%M:%%S"))) >> temp_file.txt
echo print("Python version: {}".format(sys.version)) >> temp_file.txt
echo. >> temp_file.txt
type executable_app.py >> temp_file.txt
move /y temp_file.txt executable_app.py.debug
echo [√] Created debug version of executable_app.py

echo Building debug executable (Console mode to show errors)...
pyinstaller --name=TradingBot_Debug ^
  --console ^
  --clean ^
  --noconfirm ^
  --hidden-import=flask_socketio ^
  --hidden-import=eventlet.hubs ^
  --hidden-import=dns ^
  --hidden-import=dns.resolver ^
  --hidden-import=engineio.async_drivers.threading ^
  --add-data=templates;templates ^
  --add-data=static;static ^
  executable_app.py.debug

echo Creating essential directories in distribution...
if not exist dist\model mkdir dist\model
if not exist dist\data mkdir dist\data

echo ====================================================================
echo DEBUG BUILD COMPLETED!
echo ====================================================================
echo.
echo The debug executable is available at:
echo dist\TradingBot_Debug.exe
echo.
echo This version will show any errors in a console window.
echo.
pause