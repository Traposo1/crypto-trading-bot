#!/bin/bash
# Shell script to build the Trading Bot executable

echo "==== Crypto Trading Bot Executable Builder ===="
echo "This script will build an executable version of the Trading Bot"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.7 or higher and try again"
    exit 1
fi

# Check Python version
python_version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python version: $python_version"

# Check if PyInstaller is installed
if ! python -c "import PyInstaller" &> /dev/null; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
    if [ $? -ne 0 ]; then
        echo "Failed to install PyInstaller. Please install manually with:"
        echo "pip install pyinstaller"
        exit 1
    fi
fi

# Ensure model directory exists
mkdir -p model

# Create empty model files if they don't exist
for file in "model.pkl" "scaler.pkl" "feature_list.pkl"; do
    if [ ! -f "model/$file" ]; then
        echo "Creating empty model file: $file"
        touch "model/$file"
    fi
done

# Check operating system
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows operating system"
    # Windows specific settings
    SEPARATOR=";"
else
    echo "Detected Unix-like operating system"
    # Unix specific settings
    SEPARATOR=":"
fi

# Run PyInstaller
echo "Starting build process with PyInstaller..."
echo "This may take several minutes..."

# Basic command for any platform
pyinstaller --name=TradingBot --onefile --windowed \
    --add-data="templates${SEPARATOR}templates" \
    --add-data="static${SEPARATOR}static" \
    --hidden-import=sklearn.ensemble \
    --hidden-import=sklearn.tree \
    --hidden-import=sklearn.preprocessing \
    --hidden-import=sklearn.neural_network \
    --hidden-import=flask_socketio \
    --hidden-import=eventlet \
    --hidden-import=eventlet.hubs \
    --hidden-import=dns \
    --hidden-import=dns.resolver \
    --hidden-import=engineio.async_drivers \
    --hidden-import=engineio.async_drivers.threading \
    --hidden-import=socketio \
    --hidden-import=websocket \
    --hidden-import=websocket._socket \
    --hidden-import=pandas_ta \
    --hidden-import=flask \
    --hidden-import=ccxt \
    --hidden-import=webbrowser \
    executable_app.py

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Build completed successfully!"
    echo "The executable file is in the 'dist' folder."
    echo ""
    echo "Instructions for distribution:"
    echo "1. Share the TradingBot executable from the 'dist' folder"
    echo "2. Include the 'model' folder with the executable"
    echo "3. Optionally include a config.json file for default settings"
    echo ""
else
    echo "Build failed. Check the output for errors."
    exit 1
fi