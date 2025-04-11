"""
Flask-SocketIO patch for PyInstaller compatibility
This file must be imported before any other Flask-SocketIO imports
"""
import sys
import os

# Set the environment variable first (must happen before any other imports)
os.environ['FLASK_SOCKETIO_ASYNC_MODE'] = 'threading'

def patch_flask_socketio():
    """Force flask-socketio to use threading mode by monkeypatching at runtime"""
    try:
        # First import modules where our monkey patching will take place
        import flask_socketio
        import socketio.server
        import engineio.server
        
        # Patch flask_socketio
        original_init = flask_socketio.SocketIO.__init__
        
        def patched_init(self, app=None, **kwargs):
            # Force threading mode in kwargs
            kwargs['async_mode'] = 'threading'
            # Call original init with forced threading mode
            original_init(self, app, **kwargs)
        
        flask_socketio.SocketIO.__init__ = patched_init
        
        # Patch engineio.server to always use threading
        original_server_init = engineio.server.Server.__init__
        
        def patched_server_init(self, *args, **kwargs):
            # Force threading mode
            kwargs['async_mode'] = 'threading'
            return original_server_init(self, *args, **kwargs)
            
        engineio.server.Server.__init__ = patched_server_init
        
        # Also patch socketio.server
        original_socketio_init = socketio.server.Server.__init__
        
        def patched_socketio_init(self, *args, **kwargs):
            # Force threading mode
            kwargs['async_mode'] = 'threading'
            return original_socketio_init(self, *args, **kwargs)
            
        socketio.server.Server.__init__ = patched_socketio_init
        
        print("→ Flask-SocketIO successfully patched to use threading mode")
        return True
    except Exception as e:
        print(f"→ Error patching Flask-SocketIO: {e}")
        return False

# Execute patch when this module is imported
patch_success = patch_flask_socketio()