"""
Simple direct PyInstaller build script with minimal complexity
"""
import os
import sys
import subprocess
import shutil

print("=" * 80)
print("SIMPLE DIRECT BUILD SCRIPT")
print("=" * 80)

# Check Python version
print(f"Python version: {sys.version}")

# Make sure the model directory exists
if not os.path.exists("model"):
    os.makedirs("model")
    print("Created model directory")

# Make sure dist directory exists
if not os.path.exists("dist"):
    os.makedirs("dist")
    print("Created dist directory")

# Define the direct PyInstaller command
pyinstaller_cmd = [
    "pyinstaller",
    "--name=TradingBot",
    "--onefile",  # Create a single executable file
    "--windowed",  # Hide the console when running
    "--clean",     # Clean PyInstaller cache
    "--noconfirm", # Replace previous build without asking
    "--distpath=./dist",
    "--add-data=templates:templates",
    "--add-data=static:static",
]

# Add hidden imports directly
hidden_imports = [
    "eventlet.hubs",
    "dns",
    "dns.resolver",
    "engineio.async_drivers",
    "engineio.async_drivers.threading",
    "socketio",
    "socketio.client",
    "websocket",
    "flask_socketio"
]

for imp in hidden_imports:
    pyinstaller_cmd.extend(["--hidden-import", imp])

# Add the entry point
pyinstaller_cmd.append("executable_app.py")

print(f"Running PyInstaller with command: {' '.join(pyinstaller_cmd)}")

try:
    # Run PyInstaller directly
    process = subprocess.run(pyinstaller_cmd, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
    
    if process.returncode != 0:
        print("PyInstaller failed with error:")
        print(process.stderr)
        
        # Show abbreviated output if it's too long
        if len(process.stderr) > 1000:
            print("\nAbbreviated error (last 1000 characters):")
            print(process.stderr[-1000:])
    else:
        print("PyInstaller completed successfully!")
        
        # Create model placeholders in dist folder
        dist_model_dir = os.path.join("dist", "model")
        if not os.path.exists(dist_model_dir):
            os.makedirs(dist_model_dir)
            
        # Create empty model files if they don't exist
        model_files = ["model.pkl", "scaler.pkl", "feature_list.pkl"]
        for model_file in model_files:
            model_path = os.path.join("model", model_file)
            if not os.path.exists(model_path):
                # Create empty file
                with open(model_path, 'wb') as f:
                    pass
            # Copy to dist
            shutil.copy(model_path, os.path.join(dist_model_dir, model_file))
            
        # Create data directory for SQLite
        data_dir = os.path.join("dist", "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Copy README
        if os.path.exists("README_EXECUTABLE.md"):
            shutil.copy("README_EXECUTABLE.md", os.path.join("dist", "README.md"))
            
        print("\nBuild completed successfully!")
        print(f"Executable is located at: dist/TradingBot.exe")
        
except Exception as e:
    print(f"Error running PyInstaller: {e}")

print("\nPress Enter to exit...")
input()