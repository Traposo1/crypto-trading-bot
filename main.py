# First import basic modules that don't need patching
import sys
import os
import time
import logging

# Configure logging right away
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('crypto_trader.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting application initialization")

# Import our patching module before Flask-SocketIO
try:
    import socketio_patch
    logger.info("SocketIO patch loaded successfully")
except Exception as e:
    logger.warning(f"Could not load SocketIO patch: {e}")
    
# We've already configured logging above
# Now continue with Flask and app initialization

try:
    # Import render_template first so it's available for error handlers
    from flask import render_template
    
    # Now import Flask app and socketio
    from app import app, socketio
    from bot import initialize_bot
    
    # Import and register API routes
    try:
        from routes.api import api_bp
        app.register_blueprint(api_bp, url_prefix='/api')
        logger.info("API routes registered successfully")
    except Exception as e:
        logger.error(f"Error registering API routes: {str(e)}")
        # We continue execution even if API routes fail to register
    
    # Special initialization that happens at startup
    # Modern Flask approach using with_appcontext pattern
    with app.app_context():
        try:
            logger.info("Initializing application components...")
            # Initialize the trading bot at startup
            initialize_bot()
            logger.info("App initialization complete")
        except Exception as e:
            logger.error(f"Error during app initialization: {str(e)}", exc_info=True)
            # We continue execution - we'll show error pages instead of crashing
    
    # Register error handlers for different HTTP errors
    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        return render_template('critical_error.html', title="Server Error", 
                               error=f"Internal server error: {str(e)}"), 500
    
    @app.errorhandler(404)
    def not_found(e):
        return render_template('error.html', title="Page Not Found", 
                               error="The requested page does not exist."), 404
    
    # When running this file directly with python, use socketio.run
    if __name__ == "__main__":
        try:
            logger.info("Starting application in standalone mode")
            socketio.run(app, host="0.0.0.0", port=5000, debug=True, 
                         allow_unsafe_werkzeug=True, use_reloader=True, log_output=True)
        except Exception as e:
            logger.critical(f"Failed to start application: {str(e)}", exc_info=True)
            sys.exit(1)
    # For Gunicorn, export the Flask app, SocketIO will handle connections through it
    else:
        # The Gunicorn command should be:
        # gunicorn --bind 0.0.0.0:5000 --workers=1 --worker-class eventlet --timeout 120 main:app
        try:
            logger.info("Initializing application for Gunicorn")
            # Initialize any Gunicorn-specific settings here if needed
        except Exception as e:
            logger.critical(f"Error initializing app for Gunicorn: {str(e)}", exc_info=True)
except Exception as e:
    # Catch any fatal errors during import/initialization
    try:
        # Try to log the error
        logging.critical(f"Fatal error during application startup: {str(e)}", exc_info=True)
    except:
        # If even logging fails, print to stderr as last resort
        import traceback
        print(f"CRITICAL ERROR: {str(e)}", file=sys.stderr)
        traceback.print_exc()
