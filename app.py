import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up database
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///crypto_bot.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
    "pool_timeout": 30,
    "pool_size": 10,
    "max_overflow": 15,
    "connect_args": {"timeout": 30} if "sqlite" in os.environ.get("DATABASE_URL", "sqlite://") else {}
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with extensions
db.init_app(app)

# Initialize SocketIO with ultra-high reliability settings
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    ping_timeout=180,  # 3 minute timeout for maximum reliability
    ping_interval=2.5,  # Very frequent pings (every 2.5 seconds)
    async_mode='threading',  # Explicitly use threading mode to avoid eventlet issues
    logger=True,  # Enable logging
    engineio_logger=True,  # Enable Engine.IO logging
    reconnection=True,  # Enable automatic reconnection
    max_http_buffer_size=1024 * 1024,  # 1MB buffer size for large messages
    manage_session=True,  # Manage Flask sessions properly
    always_connect=True,  # Always attempt to connect
    upgrade_timeout=10  # 10 second timeout for transport upgrades
)

# Add a simple health check endpoint for troubleshooting
@app.route('/health')
def health_check():
    """Simple health check endpoint that doesn't depend on complex app features"""
    import time
    import platform
    import sys
    
    logger.info("Health check endpoint called")
    try:
        # Test database connection
        db_status = True
        db_conn = "Unknown"
        try:
            from sqlalchemy import text
            result = db.session.execute(text("SELECT 1")).fetchone()
            db_conn = f"OK - {result[0]}"
        except Exception as e:
            db_conn = f"Error: {str(e)}"
            db_status = False
        
        # Get basic system information
        system_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Return as plain text to avoid template rendering issues
        return f"""
        CryptoTrader Pro - Health Check
        -------------------------------
        Status: OK
        Flask: Working
        Database: {db_conn}
        Python: {system_info['python_version'].split()[0]}
        Platform: {system_info['platform']}
        Time: {system_info['time']}
        """
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return f"ERROR: {str(e)}", 500

# Define a custom error handler for 500 internal server errors
@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal Server Error: {str(e)}")
    
    # Get additional system information for debugging
    import platform
    import sys
    import time
    import traceback
    
    # Get the full traceback
    error_traceback = traceback.format_exc()
    
    system_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'error': str(e),
        'traceback': error_traceback
    }
    
    # Log detailed system information
    logger.error(f"System Info: {system_info}")
    
    try:
        # First try the template
        from flask import render_template
        return render_template(
            'critical_error.html',
            title="Server Error",
            error=f"Internal Server Error: {str(e)}\n\nTraceback:\n{error_traceback}",
            now=time.localtime,
            request=None
        ), 500
    except Exception as template_error:
        # If template fails, return a simple string response
        logger.critical(f"Error template failed: {str(template_error)}")
        return f"""
        <html>
        <head><title>Critical Error</title></head>
        <body style="font-family: monospace; background: #1e1e1e; color: #ddd; padding: 20px;">
            <h1 style="color: #ff6b6b;">Critical System Error</h1>
            <div style="background: #333; padding: 10px; border-left: 5px solid #ff6b6b; white-space: pre-wrap;">
            {str(e)}
            
            {error_traceback}
            </div>
            <p>Template rendering also failed: {str(template_error)}</p>
            <a href="/health" style="color: #4dabf7; text-decoration: none;">Check System Health</a>
        </body>
        </html>
        """, 500

# Import routes after the app is created to avoid circular imports
with app.app_context():
    # Import models to ensure tables are created
    import models  # noqa: F401
    
    # Create database tables with retry mechanism
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            db.create_all()
            logger.info("Database tables created successfully")
            break
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to create database tables after {max_retries} attempts: {str(e)}")
                # Continue anyway, we can still try to run the app
            else:
                logger.warning(f"Database creation attempt {retry_count} failed: {str(e)}. Retrying...")
                # Wait before retrying (exponential backoff)
                import time
                time.sleep(0.5 * retry_count)
                
                # Rollback any failed transaction
                try:
                    db.session.rollback()
                except:
                    pass
    
    # Register blueprints
    from routes.dashboard import dashboard_bp
    from routes.trades import trades_bp
    from routes.settings import settings_bp
    from routes.backtest import backtest_bp
    
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(trades_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(backtest_bp)
    
    # Import bot components
    from bot import initialize_bot
    initialize_bot()

    # Import WebSocket events to register them
    import websocket_events  # noqa: F401

    logger.info("Crypto trading bot initialized")
