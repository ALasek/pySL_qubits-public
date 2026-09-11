################################################################################################################################
# GPU hardware info
################################################################################################################################
"""
CLI README

This module is not a standalone CLI. When imported by pySL.py it reads
sys.argv[1], if present, only to display the requested CUDA device id.
The actual user-facing CLI is documented at the top of pySL.py.
"""

import sys
import os
import pycuda
import pycuda.driver as cuda

def print_gpu_info():
    cuda.init()
    print("\n----------------------------------------------------------------")
    print("Python version: " + sys.version)
    print("pyCUDA version: " + pycuda.VERSION_TEXT)
    print("CUDA runtime version: %d.%d.%d"%cuda.get_version())
    print("----------------------------------------------------------------")
    print("CUDA capable device count:", cuda.Device.count())
    if os.environ.get("CUDA_DEVICE") is not None:
        target_device = os.environ["CUDA_DEVICE"]
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        target_device = sys.argv[1]
    else:
        target_device = "0"
    print("Target CUDA device: ", str(target_device))

    for devicenum in range(cuda.Device.count()):
        device=cuda.Device(devicenum)
        dev_name=device.name()
        dev_cc=device.compute_capability()
        dev_mem=device.total_memory()
        
        print("\n===Attributes for device %d"%devicenum, ":",dev_name)
        print("   Compute capability: %d.%d"%dev_cc)
        print("   Total global memory: %.1fGB"%(dev_mem/1073741824))

        # Uncomment the lines below for complete device diagonstics
        
        # attrs=device.get_attributes()
        # for (key,value) in attrs.items():
        #     print("%s:%s"%(str(key),str(value)))
