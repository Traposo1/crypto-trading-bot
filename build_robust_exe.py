"""
Robust PyInstaller build script with error handling and dependency management
"""
import os
import sys
import subprocess
import shutil
import time
import tempfile

# Print banner
print("="*80)
print("ROBUST TRADING BOT EXECUTABLE BUILDER")
print("="*80)

# Check Python version
print(f"Python version: {sys.version}")

# Ensure temp directory is clean
temp_dir = tempfile.mkdtemp(prefix="tradingbot_build_")
print(f"Using temporary build directory: {temp_dir}")

# Define color codes for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(message):
    print(f"\n{Colors.HEADER}[STEP] {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.FAIL}[ERROR] {message}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.WARNING}[WARNING] {message}{Colors.ENDC}")

# Ensure required directories exist
for directory in ['model', 'dist', 'build']:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Step 1: Verify and install dependencies
print_step("Installing required dependencies")

dependencies = [
    "pyinstaller>=5.6.2",
    "flask-socketio>=5.3.0",
    "python-socketio>=5.8.0",
    "python-engineio>=4.5.0",
    "eventlet>=0.33.0",
    "websocket-client>=1.5.0",
    "dnspython>=2.3.0"
]

for dependency in dependencies:
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", dependency]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        package_name = dependency.split('>=')[0]
        print(f"  ✓ Installed {package_name}")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install {dependency}")
        print(f"Error output: {e.stderr.decode() if e.stderr else 'None'}")
        sys.exit(1)

# Step 2: Clean previous builds
print_step("Cleaning previous builds")

# Try to clean up any leftover processes
if os.name == 'nt':  # Windows
    try:
        os.system('taskkill /F /IM "TradingBot.exe" /T')
        print("  ✓ Killed any running TradingBot processes")
    except:
        pass

# Delete build artifacts
for file_path in ['dist/TradingBot.exe', 'build/TradingBot']:
    if os.path.exists(file_path):
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            print(f"  ✓ Removed {file_path}")
        except Exception as e:
            print_warning(f"Could not remove {file_path}: {e}")

# Step 3: Create model placeholders if they don't exist
print_step("Setting up model files")

model_files = ["model.pkl", "scaler.pkl", "feature_list.pkl"]
for model_file in model_files:
    model_path = os.path.join("model", model_file)
    if not os.path.exists(model_path):
        # Create empty file
        with open(model_path, 'wb') as f:
            pass
        print(f"  ✓ Created empty placeholder for {model_file}")

# Step 4: Generate PyInstaller command
print_step("Configuring PyInstaller")

# Use a two-phase approach to avoid common PyInstaller issues
# First generate a spec file, then build from it
spec_file = os.path.join(temp_dir, "TradingBot.spec")

hidden_imports = [
    "eventlet.hubs",
    "eventlet.hubs.selects",
    "eventlet.hubs.epolls", 
    "eventlet.hubs.kqueues",
    "eventlet.greenpool",
    "eventlet.semaphore",
    "eventlet.queue",
    "dns",
    "dns.resolver",
    "engineio.async_drivers",
    "engineio.async_drivers.threading",
    "socketio", 
    "socketio.client",
    "socketio.server",
    "websocket",
    "websocket._socket",
    "flask_socketio",
    "sqlalchemy.sql.default_comparator"
]

sklearn_imports = [
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.ensemble._forest",
    "sklearn.ensemble._gb",
    "sklearn.ensemble._weight_boosting",
    "sklearn.neural_network._multilayer_perceptron",
    "sklearn.utils._weight_vector",
    "sklearn.pipeline",
    "sklearn.neural_network.multilayer_perceptron",
    "sklearn.ensemble.forest",
    "sklearn.ensemble.gradient_boosting",
    "sklearn.ensemble.weight_boosting",
    "sklearn.metrics.classification",
    "sklearn.decomposition.pca",
    "numpy.random",
    "pandas",
    "pandas._libs",
    "pandas._libs.algos",
]

hidden_imports.extend(sklearn_imports)

# Step 5: Create a spec file first
print_step("Generating spec file")

pyi_makespec_cmd = [
    "pyinstaller",
    "--name=TradingBot",
    "--onefile",  # Create a single executable file
    "--windowed",  # Hide console when running
    "--specpath", temp_dir,
    "--workpath", os.path.join(temp_dir, "build"),
    "--distpath", "dist",
    "--add-data=templates:templates",  # Include template files
    "--add-data=static:static",        # Include static files
]

# Add all hidden imports to the command
for imp in hidden_imports:
    pyi_makespec_cmd.extend(["--hidden-import", imp])

# Add the entry point script
pyi_makespec_cmd.append("executable_app.py")

# Run the makespec command
print(f"Running: {' '.join(pyi_makespec_cmd)}")
try:
    subprocess.run(pyi_makespec_cmd, check=True)
    print_success("Spec file generated successfully")
except subprocess.CalledProcessError as e:
    print_error(f"Failed to generate spec file: {e}")
    sys.exit(1)

# Step 6: Modify the spec file to enhance compatibility
print_step("Enhancing spec file configuration")

if os.path.exists(spec_file):
    with open(spec_file, 'r') as f:
        spec_content = f.read()
    
    # Add runtime hooks to ensure proper initialization
    runtime_hooks_code = """
# Add runtime hooks for better Flask-SocketIO compatibility
a.runtime_hooks.append('build_hooks.py')
    """
    
    # Insert before the EXE definition
    if "EXE(" in spec_content:
        spec_content = spec_content.replace("EXE(", runtime_hooks_code + "\nEXE(")
        
        # Write out the modified spec
        with open(spec_file, 'w') as f:
            f.write(spec_content)
        print("  ✓ Enhanced spec file with runtime hooks")
    else:
        print_warning("Could not find EXE() in spec file, skipping hooks addition")

# Create a runtime hook file
hook_file = 'build_hooks.py'
with open(hook_file, 'w') as f:
    f.write("""
# Runtime hooks for PyInstaller
import os
import sys

# Allow auto-detection of async mode
if 'FLASK_SOCKETIO_ASYNC_MODE' in os.environ:
    del os.environ['FLASK_SOCKETIO_ASYNC_MODE']

# Ensure imports work properly
sys.path.insert(0, os.path.dirname(sys.executable))
""")
print("  ✓ Created runtime hooks file")

# Step 7: Now build from the spec file
print_step("Building executable from spec")

build_cmd = [
    "pyinstaller",
    "--clean",  # Clean PyInstaller cache before building
    spec_file
]

try:
    print(f"Running: {' '.join(build_cmd)}")
    process = subprocess.run(build_cmd, capture_output=True, text=True)
    
    # Check if the executable was created
    exe_path = os.path.join("dist", "TradingBot.exe") if os.name == 'nt' else os.path.join("dist", "TradingBot")
    if os.path.exists(exe_path):
        print_success(f"Executable built successfully: {exe_path}")
    else:
        print_error("Build command ran, but executable was not created")
        print(f"Output: {process.stdout}")
        print(f"Error: {process.stderr}")
        sys.exit(1)
except subprocess.CalledProcessError as e:
    print_error(f"Build failed with error code: {e.returncode}")
    print(f"Error output: {e.stderr}")
    sys.exit(1)

# Step 8: Copy additional necessary files to dist
print_step("Preparing distribution package")

# Create model directory in dist if it doesn't exist
dist_model_dir = os.path.join("dist", "model")
if not os.path.exists(dist_model_dir):
    os.makedirs(dist_model_dir)
    print(f"  ✓ Created model directory in dist")

# Copy model files
for model_file in model_files:
    src_path = os.path.join("model", model_file)
    dst_path = os.path.join(dist_model_dir, model_file)
    shutil.copy2(src_path, dst_path)
    print(f"  ✓ Copied {model_file} to distribution")

# Copy documentation
shutil.copy2("README_EXECUTABLE.md", os.path.join("dist", "README.md"))
print("  ✓ Copied README to distribution")

# Create data directory in dist for SQLite database
data_dir = os.path.join("dist", "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"  ✓ Created data directory in dist")

# Step 9: Clean up temporary files
print_step("Cleaning up")
try:
    shutil.rmtree(temp_dir)
    os.remove(hook_file)
    print("  ✓ Removed temporary build files")
except Exception as e:
    print_warning(f"Could not clean up some temporary files: {e}")

# Final message
print("\n" + "="*80)
print_success("BUILD COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nExecutable is located at: {exe_path}")
print("\nDistribution package contains:")
print("  - TradingBot executable")
print("  - model/ directory with model files")
print("  - data/ directory for database storage")
print("  - README.md with instructions")
print("\nTo distribute, simply zip the 'dist' folder and share it.")
print("\nPress Enter to exit...")
input()