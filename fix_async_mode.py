"""
Direct fix for the async_mode issue in Flask-SocketIO with PyInstaller
"""
import os
import sys
import re
import glob
import shutil
import tempfile
import subprocess
from pathlib import Path

print("="*80)
print("FLASK-SOCKETIO ASYNC MODE FIX UTILITY")
print("="*80)

# Function to find Python's site-packages directory
def find_site_packages():
    """Find the site-packages directory for the current Python environment"""
    for path in sys.path:
        if path.endswith('site-packages'):
            return path
    return None

# Function to patch Flask-SocketIO files
def patch_socketio_files():
    """Patch Flask-SocketIO files to fix async_mode issues with PyInstaller"""
    site_packages = find_site_packages()
    if not site_packages:
        print("ERROR: Could not find site-packages directory")
        return False
    
    print(f"Found site-packages at: {site_packages}")
    
    # Files to patch
    flask_socketio_init = os.path.join(site_packages, 'flask_socketio', '__init__.py')
    socketio_server = os.path.join(site_packages, 'socketio', 'server.py')
    
    files_to_check = [flask_socketio_init, socketio_server]
    for file in files_to_check:
        if not os.path.exists(file):
            print(f"WARNING: File {file} not found")
    
    patched = False
    
    # Patch flask_socketio/__init__.py
    if os.path.exists(flask_socketio_init):
        print(f"Patching {flask_socketio_init}")
        with open(flask_socketio_init, 'r') as f:
            content = f.read()
        
        # Make a backup
        backup_file = flask_socketio_init + '.bak'
        if not os.path.exists(backup_file):
            shutil.copy2(flask_socketio_init, backup_file)
            print(f"Created backup at {backup_file}")
        
        # Replace async_mode detection code
        original_pattern = r"def __init__\(self, app=None, \*\*kwargs\):(.*?)async_mode = kwargs\.pop\('async_mode', None\)(.*?)if async_mode == 'eventlet':"
        replacement = r"def __init__(self, app=None, **kwargs):\1async_mode = kwargs.pop('async_mode', 'threading')\2if async_mode == 'eventlet':"
        
        new_content = re.sub(original_pattern, replacement, content, flags=re.DOTALL)
        
        if new_content != content:
            with open(flask_socketio_init, 'w') as f:
                f.write(new_content)
            print("  Successfully patched flask_socketio/__init__.py")
            patched = True
        else:
            print("  No changes needed or pattern not found in flask_socketio/__init__.py")
    
    # Create a patch file for runtime
    runtime_patch = """
# Runtime patch for Flask-SocketIO with PyInstaller
# This modifies Flask-SocketIO's behavior at runtime to force 'threading' mode
import sys
import os

def patch_flask_socketio():
    """Force flask-socketio to use threading mode"""
    try:
        import flask_socketio
        flask_socketio.SocketIO._async_mode = 'threading'
        flask_socketio.SocketIO._async_handlers = False
        
        # Also patch engineio.server
        import engineio.server
        original_init = engineio.server.Server.__init__
        
        def patched_init(self, async_mode='threading', *args, **kwargs):
            if async_mode is None:
                async_mode = 'threading'
            return original_init(self, async_mode=async_mode, *args, **kwargs)
        
        engineio.server.Server.__init__ = patched_init
        
        print("Successfully patched Flask-SocketIO to use threading mode")
    except Exception as e:
        print(f"Error patching Flask-SocketIO: {e}")

patch_flask_socketio()
"""
    
    with open('flask_socketio_patch.py', 'w') as f:
        f.write(runtime_patch)
    print("Created flask_socketio_patch.py for runtime patching")
    
    # Create PyInstaller hook file
    hook_content = """
# PyInstaller hook for Flask-SocketIO to ensure proper initialization
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Add the patch to the import list
hidden_imports = ['flask_socketio_patch']

# Add engine.io and socket.io modules
hidden_imports.extend(collect_submodules('engineio'))
hidden_imports.extend(collect_submodules('socketio'))
hidden_imports.extend(['dns', 'dns.resolver', 'eventlet.hubs'])

# Add data files
datas = collect_data_files('flask_socketio', include_py_files=True)
datas += collect_data_files('engineio', include_py_files=True)
datas += collect_data_files('socketio', include_py_files=True)
"""
    
    with open('hook-flask_socketio.py', 'w') as f:
        f.write(hook_content)
    print("Created hook-flask_socketio.py for PyInstaller")
    
    # Create a simple build script
    build_script = """
import subprocess
import os
import sys

# Ensure the patch is imported first
with open("executable_app.py", "r") as f:
    content = f.read()

if "import flask_socketio_patch" not in content:
    with open("executable_app.py", "r") as f:
        lines = f.readlines()
    
    # Find the first import statement
    import_index = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            import_index = i
            break
    
    # Insert our patch import
    lines.insert(import_index, "import flask_socketio_patch  # Patch to fix Flask-SocketIO in PyInstaller\\n")
    
    with open("executable_app.py", "w") as f:
        f.writelines(lines)
    print("Added flask_socketio_patch import to executable_app.py")

# Create a direct application with threading mode
cmd = [
    "pyinstaller",
    "--name=TradingBot",
    "--onefile",
    "--windowed",
    "--clean",
    "--additional-hooks-dir=.",
    "--hidden-import=flask_socketio_patch",
    "--hidden-import=engineio.async_drivers.threading",
    "--hidden-import=eventlet.hubs",
    "--hidden-import=dns",
    "--hidden-import=dns.resolver",
    "--add-data=templates;templates",
    "--add-data=static;static",
    "executable_app.py"
]

print("Running PyInstaller with command:")
print(" ".join(cmd))

result = subprocess.run(cmd)
if result.returncode != 0:
    print("PyInstaller failed")
    sys.exit(1)

# Create model directory if it doesn't exist
os.makedirs("dist/model", exist_ok=True)
os.makedirs("dist/data", exist_ok=True)

# Copy README if it exists
if os.path.exists("README_EXECUTABLE.md"):
    import shutil
    shutil.copy("README_EXECUTABLE.md", "dist/README.md")

print("Build completed successfully!")
"""
    
    with open('build_fixed_executable.py', 'w') as f:
        f.write(build_script)
    print("Created build_fixed_executable.py script")
    
    return patched

# Create a Windows batch file for easier execution
def create_batch_file():
    """Create a Windows batch file to run the fix and build process"""
    batch_content = """@echo off
echo ============================================================
echo FLASK-SOCKETIO ASYNC MODE FIX AND BUILD UTILITY
echo ============================================================

echo Installing required packages...
pip install pyinstaller flask-socketio eventlet dnspython

echo Applying Flask-SocketIO async_mode fix...
python fix_async_mode.py

echo Building the executable...
python build_fixed_executable.py

echo ============================================================
echo Process completed!
echo Check the dist directory for the executable.
echo ============================================================

pause
"""
    
    with open('fix_and_build.bat', 'w') as f:
        f.write(batch_content)
    print("Created fix_and_build.bat for Windows users")

# Main function
def main():
    """Main function"""
    # Patch the Flask-SocketIO files
    patched = patch_socketio_files()
    
    # Create the batch file
    create_batch_file()
    
    print("\nNext steps:")
    print("1. Run 'fix_and_build.bat' (Windows) or 'python build_fixed_executable.py' (all platforms)")
    print("2. The fixed executable will be in the dist directory")
    print("3. If issues persist, check logs in the logs directory after running the executable")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())