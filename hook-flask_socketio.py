"""
PyInstaller hook to properly include Flask-SocketIO dependencies
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Include all modules from these packages
hiddenimports = []
hiddenimports.extend(collect_submodules('engineio'))
hiddenimports.extend(collect_submodules('socketio'))
hiddenimports.extend(collect_submodules('flask_socketio'))
hiddenimports.extend(['eventlet', 'eventlet.hubs', 'dns', 'dns.resolver'])

# Force threading mode for eventlet
def pre_find_module_path(hook_api):
    # This will be executed when the hook is loaded
    import os
    os.environ['EVENTLET_THREADPOOL_SIZE'] = '20'
    os.environ['FLASK_SOCKETIO_ASYNC_MODE'] = 'threading'