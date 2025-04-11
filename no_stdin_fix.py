"""
PyInstaller hook to fix input/stdin issues in executables
This is a common issue with PyInstaller executables when stdin is not available
"""
import os
import sys

# Suppress all stdin-related errors
def redirect_stdin():
    try:
        sys.stdin = open(os.devnull, 'r')
        print("[✓] Successfully redirected stdin to prevent PyInstaller stdin errors")
    except Exception as e:
        print(f"[!] Error setting up stdin redirection: {e}")
        # Fallback method if the first approach fails
        try:
            fd = os.open(os.devnull, os.O_RDONLY)
            os.dup2(fd, 0)  # 0 is the file descriptor for stdin
            os.close(fd)
            print("[✓] Successfully redirected stdin using file descriptors")
        except Exception as e:
            print(f"[!] Failed to redirect stdin with fallback method: {e}")

# Execute immediately
redirect_stdin()