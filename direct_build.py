"""
Super direct build script that works on most systems
"""
import os
import sys
import subprocess
import shutil
import tempfile

# Banner
print("="*80)
print("DIRECT EXECUTABLE BUILD SCRIPT")
print("="*80)

# Create necessary directories 
os.makedirs('dist', exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('build', exist_ok=True)

# Check if executable_app.py exists
if not os.path.exists('executable_app.py'):
    print("Error: executable_app.py not found")
    sys.exit(1)

# Create a console script that will show any errors
console_script = """
# This file is auto-generated for debugging PyInstaller issues
import os
import sys
import io
import time
import ctypes
import traceback

# Try to unblock stdio 
if hasattr(ctypes, 'windll'):
    k = ctypes.windll.kernel32
    k.SetStdHandle(-11, k._get_osfhandle(sys.stdout.fileno()))

# Set up a log file
log_file = open('debug.log', 'w')

# Original print function
original_print = print

# Override print to write to log file too
def print_both(*args, **kwargs):
    # Get the string representation
    output_string = ' '.join(str(arg) for arg in args)
    
    # Call original print
    original_print(*args, **kwargs)
    
    # Write to log file
    log_file.write(output_string + '\\n')
    log_file.flush()

# Replace the built-in print function
print = print_both

print("="*80)
print("TRADING BOT DEBUG LOG")
print("="*80)
print("Debug log started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("Python version:", sys.version)
print("Platform:", sys.platform)
print("-"*80)

# Check if critical modules are available
try:
    print("Checking for Flask...")
    import flask
    print("[OK] Flask available:", flask.__version__)
except Exception as e:
    print("[ERROR] Flask import error:", str(e))

try:
    print("Checking for Flask-SocketIO...")
    import flask_socketio
    print("[OK] Flask-SocketIO available:", flask_socketio.__version__)
except Exception as e:
    print("[ERROR] Flask-SocketIO import error:", str(e))

try:
    print("Checking for SQLAlchemy...")
    import sqlalchemy
    print("[OK] SQLAlchemy available:", sqlalchemy.__version__)
except Exception as e:
    print("[ERROR] SQLAlchemy import error:", str(e))

print("-"*80)
print("Starting application...")

try:
    # Wrapped in try-except to catch all possible errors
    try:
        # Import the socket fix first
        print("Importing flask_socketio_patch...")
        try:
            import flask_socketio_patch
            print("[OK] Socket patch imported successfully")
        except Exception as e:
            print("[ERROR] Socket patch import failed:", str(e))
        
        # Import the stdin fix
        print("Importing no_stdin_fix...")
        try:
            import no_stdin_fix
            print("[OK] Stdin fix imported successfully")
        except Exception as e:
            print("[ERROR] Stdin fix import failed:", str(e))
            
        # Then import the main app
        print("Importing executable_app...")
        import executable_app
        print("[OK] App imported successfully")
        
        # Run the app
        print("Starting app.run_app()...")
        executable_app.run_app()
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        print(f"Failed to import required module. Check if all dependencies are installed.")
        traceback.print_exc()
    except AttributeError as e:
        print(f"ATTRIBUTE ERROR: {e}")
        print(f"An object doesn't have the expected attribute or method.")
        traceback.print_exc()
    except Exception as e:
        print(f"GENERAL ERROR: {e}")
        traceback.print_exc()
except Exception as e:
    print(f"CRITICAL OUTER ERROR: {e}")
    traceback.print_exc()

print("-"*80)
print("Application exited or crashed. See the errors above.")
print("Press Enter to close this window...")
input()
log_file.close()
"""

# Write to a file
with open('debug_runner.py', 'w') as f:
    f.write(console_script)

print("Created debug_runner.py")

# Run PyInstaller directly
pyinstaller_cmd = [
    "pyinstaller",
    "--name=TradingBot_Console",
    "--console",  # Show console for debugging
    "--clean",
    "--noconfirm",
    "--hidden-import=flask_socketio",
    "--hidden-import=eventlet.hubs",
    "--hidden-import=dns",
    "--hidden-import=dns.resolver",
    "--hidden-import=engineio.async_drivers.threading",
    "--add-data=flask_socketio_patch.py;." if sys.platform == 'win32' else "--add-data=flask_socketio_patch.py:.",
    "--add-data=no_stdin_fix.py;." if sys.platform == 'win32' else "--add-data=no_stdin_fix.py:.",
    "--add-data=templates;templates" if sys.platform == 'win32' else "--add-data=templates:templates",
    "--add-data=static;static" if sys.platform == 'win32' else "--add-data=static:static",
    "debug_runner.py"
]

print("Running PyInstaller with command:")
print(" ".join(pyinstaller_cmd))

try:
    subprocess.run(pyinstaller_cmd, check=True)
    print("\nBuild completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"PyInstaller failed with error: {e}")
    sys.exit(1)

# Create dist directories
os.makedirs('dist/model', exist_ok=True)
os.makedirs('dist/data', exist_ok=True)

print("\nDebug executable is in the dist folder")
print("This version will show any errors in a console window")
print("If it crashes, check the debug.log file in the same directory")

input("\nPress Enter to exit...")