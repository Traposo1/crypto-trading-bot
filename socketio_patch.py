"""
Runtime patching for Flask-SocketIO 
This file provides patches for Flask-SocketIO to work reliably with both eventlet and threading modes
"""

def patch_socketio():
    """Apply runtime patches to Flask-SocketIO for more reliable operation"""
    
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Try to patch socketio
        import flask_socketio
        logger.info("Patching Flask-SocketIO for better compatibility")
        
        # Store the original run method
        original_run = flask_socketio.SocketIO.run
        
        # Define a patched run method that handles more cases gracefully
        def patched_run(self, app, host=None, port=None, **kwargs):
            logger.info(f"Running socketio with mode: {self.async_mode}")
            
            # Force eventlet to be properly initialized if we're using it
            if getattr(self, 'async_mode', None) == 'eventlet':
                try:
                    import eventlet
                    eventlet.monkey_patch()
                except ImportError:
                    logger.warning("Eventlet not available, falling back to threading mode")
                    self.async_mode = 'threading'
                    
            # Call the original run method
            return original_run(self, app, host, port, **kwargs)
        
        # Apply the patch
        flask_socketio.SocketIO.run = patched_run
        logger.info(f"Flask-SocketIO patched successfully")
        
        return True
    except Exception as e:
        logger.error(f"Failed to patch Flask-SocketIO: {str(e)}")
        return False

# Apply patches when this module is imported
patch_successful = patch_socketio()