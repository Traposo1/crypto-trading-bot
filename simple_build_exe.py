import os
import subprocess

# Basic PyInstaller command for simpler build (less imports, faster build time)
pyinstaller_cmd = [
    'pyinstaller',
    '--name=TradingBot',
    '--onefile',  # Create a single executable file
    '--windowed',  # Hide the console when running the app
    '--add-data=templates:templates',
    '--add-data=static:static',
    '--hidden-import=flask_socketio',
    '--hidden-import=eventlet',
    '--hidden-import=eventlet.hubs',
    '--hidden-import=engineio.async_drivers',
    '--hidden-import=engineio.async_drivers.threading',
    '--hidden-import=socketio',
    '--hidden-import=websocket',
    'executable_app.py'
]

# Run PyInstaller
print("Running PyInstaller with command:", ' '.join(pyinstaller_cmd))
result = subprocess.run(pyinstaller_cmd)

if result.returncode == 0:
    print("\nBuild completed successfully!")
    print("Executable is located at: dist/TradingBot.exe")
else:
    print("Build failed with error code:", result.returncode)