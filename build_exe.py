import os
import sys
import shutil
import subprocess

print("Starting to build the Trading Bot executable...")

# Make sure the model directory exists
if not os.path.exists("model"):
    os.makedirs("model")

# Default PyInstaller command for basic build
pyinstaller_cmd = [
    'pyinstaller',
    '--name=TradingBot',
    '--onefile',  # Create a single executable file
    '--windowed',  # Hide the console when running the app (for GUI applications)
    '--add-data=templates:templates',  # Include the templates directory
    '--add-data=static:static',  # Include the static directory
    '--hidden-import=sklearn.ensemble',
    '--hidden-import=sklearn.tree',
    '--hidden-import=sklearn.preprocessing',
    '--hidden-import=sklearn.neural_network',
    '--hidden-import=flask_socketio',
    '--hidden-import=eventlet',
    '--hidden-import=eventlet.hubs',
    '--hidden-import=dns',
    '--hidden-import=dns.resolver',
    '--hidden-import=engineio.async_drivers',
    '--hidden-import=engineio.async_drivers.threading',
    '--hidden-import=socketio',
    '--hidden-import=websocket',
    '--hidden-import=websocket._socket',
    '--hidden-import=pandas_ta',
    '--hidden-import=flask',
    '--hidden-import=ccxt',
    '--hidden-import=webbrowser',
    '--icon=generated-icon.png',  # Add icon if the file exists
    'executable_app.py'  # Use our specialized executable script
]

# Add additional hidden imports for all scikit-learn models
sklearn_imports = [
    'sklearn.metrics',
    'sklearn.model_selection',
    'sklearn.ensemble._forest',
    'sklearn.ensemble._gb',
    'sklearn.ensemble._weight_boosting',
    'sklearn.neural_network._multilayer_perceptron',
    'sklearn.utils._weight_vector',
    'sklearn.pipeline',
    'sklearn.neural_network.multilayer_perceptron',
    'sklearn.ensemble.forest',
    'sklearn.ensemble.gradient_boosting',
    'sklearn.ensemble.weight_boosting',
    'sklearn.metrics.classification',
    'sklearn.decomposition.pca',
    'numpy.random',
    'pandas',
    'pandas._libs',
    'pandas._libs.algos',
]

for imp in sklearn_imports:
    pyinstaller_cmd.extend(['--hidden-import', imp])

# Execute PyInstaller
print("Running PyInstaller with command:", ' '.join(pyinstaller_cmd))
process = subprocess.run(pyinstaller_cmd, capture_output=True, text=True)

if process.returncode != 0:
    print("PyInstaller failed with error:")
    print(process.stderr)
    sys.exit(1)
else:
    print("PyInstaller completed successfully!")
    print(process.stdout)

# Copy additional necessary files
print("Copying additional files to the distribution...")

# Create directory for the model in the dist folder
if not os.path.exists(os.path.join("dist", "model")):
    os.makedirs(os.path.join("dist", "model"))

# Copy model files if they exist (optional, create empty files if they don't exist)
model_files = ["model.pkl", "scaler.pkl", "feature_list.pkl"]
for model_file in model_files:
    model_path = os.path.join("model", model_file)
    if not os.path.exists(model_path):
        # Create empty file
        with open(model_path, 'wb') as f:
            pass
    shutil.copy(model_path, os.path.join("dist", "model", model_file))

# Copy configuration files
if os.path.exists("config.json"):
    shutil.copy("config.json", "dist")
else:
    # Create default config.json
    with open(os.path.join("dist", "config.json"), 'w') as f:
        f.write('{"trading_pair": "BTC/USDT", "timeframe": "5m", "paper_trading": true}')

print("\nBuild completed successfully!")
print("Executable is located at: dist/TradingBot.exe")
print("To run the application, please execute dist/TradingBot.exe")