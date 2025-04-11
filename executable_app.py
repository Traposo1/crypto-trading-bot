import os
import sys
import logging
import time
import threading
import webbrowser
import sqlite3
from datetime import datetime

# Set a specific async mode for Flask-SocketIO that works in PyInstaller
# This needs to be set before importing Flask or SocketIO
os.environ['FLASK_SOCKETIO_ASYNC_MODE'] = 'threading'

# Configure logging before anything else
logging_folder = "logs"
if not os.path.exists(logging_folder):
    os.makedirs(logging_folder)

log_file = os.path.join(logging_folder, f"tradingbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Fix resource paths for PyInstaller
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Ensure the model directory exists
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Setup SQLite database for standalone mode
def setup_local_database():
    """Setup a local SQLite database for the standalone application"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    db_path = os.path.join(data_dir, "tradingbot.db")
    db_uri = f"sqlite:///{db_path}"
    
    # Set environment variable for database connection
    os.environ["DATABASE_URL"] = db_uri
    
    print(f"Using local SQLite database at {db_path}")

# Welcome message
print("="*80)
print("Crypto Trading Bot - Starting up...")
print("="*80)

# Prevent sys.stdin errors in PyInstaller
sys.stdin = open(os.devnull, 'r')

# Import the main application after path setup
try:
    from app import app
    from bot import initialize_bot, start_bot, stop_bot
    import main
    print("Successfully imported application modules.")
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("This may indicate missing dependencies or incorrect installation.")
    try:
        input("Press Enter to exit...")
    except:
        time.sleep(5)  # Wait 5 seconds instead if input fails
    sys.exit(1)

def open_browser():
    """Open web browser after a short delay"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open('http://127.0.0.1:5000')
    print("Web browser opened to trading bot interface.")

# Main application function
def run_app():
    try:
        # Setup local database
        setup_local_database()
        
        # Check for API keys in environment variables or prompt user if needed
        if not os.environ.get("BINANCE_API_KEY") or not os.environ.get("BINANCE_API_SECRET"):
            print("Warning: No API keys found. Running in read-only mode.")
            print("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
            print("or configure them in the web interface to enable trading.")
        
        # Initialize the trading bot
        print("Initializing the trading bot...")
        initialize_bot()
        
        # Start the bot
        print("Starting the trading bot...")
        start_bot()
        
        # Open browser in a separate thread
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run the Flask application
        print("Starting web server on http://127.0.0.1:5000")
        
        # Check if socketio is available and use it
        try:
            from flask_socketio import SocketIO
            from app import socketio
            
            # Force a specific async mode for the executable
            # This overrides any automatic detection
            socketio._async_mode = 'threading'
            socketio._async_handlers = False
            
            socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, 
                        log_output=False, allow_unsafe_werkzeug=True)
        except ImportError:
            # Fall back to standard Flask if SocketIO is not available
            app.run(host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"Error running the application: {e}")
        logging.error(f"Application error: {e}", exc_info=True)
    finally:
        # Stop the bot when the application exits
        stop_bot()
        print("Trading bot stopped.")

# Entry point
if __name__ == "__main__":
    run_app()