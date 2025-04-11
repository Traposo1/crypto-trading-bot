"""
Fix for DNS module import errors in PyInstaller
This creates a dummy 'dns' package structure for PyInstaller to include
"""
import os
import sys

def create_dns_module():
    """Create a dummy dns module structure for PyInstaller compatibility"""
    try:
        # Try to find dns package
        import dns
        dns_path = os.path.dirname(dns.__file__)
        
        # Create the dnssec module if it doesn't exist
        dnssec_path = os.path.join(dns_path, 'dnssec.py')
        if not os.path.exists(dnssec_path):
            with open(dnssec_path, 'w') as f:
                f.write('"""Dummy dnssec module for PyInstaller compatibility"""\n')
                f.write('# This is a placeholder to satisfy imports\n')
                f.write('def validate(message):\n')
                f.write('    return False\n')
            print(f"[OK] Created dummy dns.dnssec module")
        else:
            print(f"[INFO] dns.dnssec module already exists")
            
        # Create the e164 module if it doesn't exist
        e164_path = os.path.join(dns_path, 'e164.py')
        if not os.path.exists(e164_path):
            with open(e164_path, 'w') as f:
                f.write('"""Dummy e164 module for PyInstaller compatibility"""\n')
                f.write('# This is a placeholder to satisfy imports\n')
                f.write('def query(number):\n')
                f.write('    return None\n')
            print(f"[OK] Created dummy dns.e164 module")
        else:
            print(f"[INFO] dns.e164 module already exists")
            
        # Create the namedict module if it doesn't exist
        namedict_path = os.path.join(dns_path, 'namedict.py')
        if not os.path.exists(namedict_path):
            with open(namedict_path, 'w') as f:
                f.write('"""Dummy namedict module for PyInstaller compatibility"""\n')
                f.write('# This is a placeholder to satisfy imports\n')
                f.write('def from_text(text):\n')
                f.write('    return {}\n')
            print(f"[OK] Created dummy dns.namedict module")
        else:
            print(f"[INFO] dns.namedict module already exists")
            
        return True
    except ImportError:
        print("[ERROR] dns package not found. Try installing dnspython first: pip install dnspython")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to create dummy dns.dnssec module: {e}")
        return False

if __name__ == "__main__":
    if create_dns_module():
        print("[SUCCESS] DNS modules fixed for PyInstaller compatibility")
    else:
        print("[WARNING] DNS modules could not be fixed")