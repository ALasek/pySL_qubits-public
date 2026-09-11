import platform
import os

def get_ccbin():

    # Default Linux compiler
    ccbin = 'g++'
    
    # Compatibility with Windows
    if os.name == 'nt': 
        ccbin = 'cl.exe'

    return ccbin
