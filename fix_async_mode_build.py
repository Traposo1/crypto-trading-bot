import os
import subprocess
import sys

print("=" * 80)
print("Specialized Build Script to Fix Async Mode Issue")
print("=" * 80)

# Ensure dependencies are installed
required_packages = [
    "flask-socketio", "eventlet", "python-socketio", "python-engineio", 
    "websocket-client", "dnspython", "bidict", "simple-websocket"
]

# Try to install required packages
print("Installing required packages...")
for package in required_packages:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package], 
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"  - {package} installed/upgraded successfully")
    except subprocess.CalledProcessError:
        print(f"  - Warning: Failed to install {package}")

# Create a basic PyInstaller command that specifically addresses the async_mode issue
pyinstaller_cmd = [
    'pyinstaller',
    '--name=TradingBot',
    '--onefile',
    '--windowed',
    '--add-data=templates:templates',
    '--add-data=static:static',
    '--hidden-import=flask_socketio',
    '--hidden-import=eventlet',
    '--hidden-import=eventlet.hubs',
    '--hidden-import=eventlet.greenpool',
    '--hidden-import=eventlet.semaphore',
    '--hidden-import=eventlet.queue',
    '--hidden-import=dns',
    '--hidden-import=dns.resolver',
    '--hidden-import=engineio.async_drivers',
    '--hidden-import=engineio.async_drivers.threading',
    '--hidden-import=socketio',
    '--hidden-import=socketio.client',
    '--hidden-import=socketio.server',
    '--hidden-import=websocket',
    '--hidden-import=websocket._socket',
    'executable_app.py'
]

# Ensure model directory exists
if not os.path.exists("model"):
    os.makedirs("model")
    print("Created model directory")

# Run PyInstaller with the fixed command
print("\nRunning PyInstaller with fixed dependencies...")
print("This may take several minutes. Please be patient.\n")

result = subprocess.run(pyinstaller_cmd)

if result.returncode == 0:
    print("\n" + "=" * 80)
    print("Build completed successfully!")
    print("=" * 80)
    print("\nExecutable is located at: dist/TradingBot.exe")
    print("\nIMPORTANT: This build includes specific fixes for the async_mode issue.")
    print("           Make sure to include the 'model' folder with your distribution.")
else:
    print("\n" + "=" * 80)
    print("Build failed with error code:", result.returncode)
    print("=" * 80)
    print("\nPlease try the following:")
    print("1. Make sure you have the latest versions of all dependencies")
    print("2. Try running with administrator privileges")
    print("3. Check for any antivirus software that might be blocking the build process")

print("\nPress Enter to exit...")
input()