"""
Fix for eventlet.hubs missing modules import error in PyInstaller
This creates simple dummy modules that PyInstaller can bundle
"""
import os
import sys

def create_dummy_module(module_name):
    """Create a dummy module for PyInstaller to include"""
    # Only needed on Windows
    if sys.platform != 'win32':
        return
        
    try:
        # Try to find eventlet package
        import eventlet
        eventlet_path = os.path.dirname(eventlet.__file__)
        
        # Path to hubs directory
        hubs_path = os.path.join(eventlet_path, 'hubs')
        
        # Check if module already exists
        module_path = os.path.join(hubs_path, f'{module_name}.py')
        if not os.path.exists(module_path):
            # Create a dummy module
            with open(module_path, 'w') as f:
                f.write(f'"""Dummy {module_name} module for PyInstaller compatibility"""\n')
                f.write('# This is a placeholder to satisfy imports\n')
                f.write('def is_available():\n')
                f.write('    return False\n')
            
            print(f"[OK] Created dummy {module_name}.py module for PyInstaller compatibility")
            return True
        else:
            print(f"[INFO] Module {module_name}.py already exists")
            return True
    except Exception as e:
        print(f"[ERROR] Failed to create dummy {module_name} module: {e}")
        return False

def fix_eventlet_modules():
    """Fix all eventlet hub modules that cause issues on Windows"""
    # List of eventlet.hubs modules that cause problems on Windows
    problematic_modules = ['epolls', 'kqueue', 'poll', 'selects']
    
    success = True
    for module in problematic_modules:
        if not create_dummy_module(module):
            success = False
    
    return success

if __name__ == "__main__":
    if fix_eventlet_modules():
        print("[SUCCESS] All eventlet hub modules fixed for PyInstaller compatibility")
    else:
        print("[WARNING] Some eventlet modules could not be fixed")