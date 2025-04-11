from main import app
import os

# Import the necessary modules
import bot
from app import socketio  # This will import the socketio instance from app.py

# Initialize the trading bot
bot.initialize_bot()

if __name__ == "__main__":
    # Use the socketio instance with proper parameters
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True, 
        use_reloader=False, 
        log_output=True,
        allow_unsafe_werkzeug=True
    )