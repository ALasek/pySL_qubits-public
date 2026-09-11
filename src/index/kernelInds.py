################################################################################################################################
# Create ND index lookup
################################################################################################################################

# This function returns an array that maps a 1D index of the wave function to the indices for each dimension corresponding to that point. 
# It is done for 1 particle only here, since for many particles it's relatively cheap to calculate the indices from that

import numpy as np
import src.index.index1DtoND as index1DtoND
import pycuda.driver as cuda

class kernelInds:
    
    def __init__(self, params):
        Ksizes = params["TDSESolver"]=="kSpace"
        
        N1 = params["Kx"] * params["Ky"] * params["Kz"] if Ksizes else params["N1"]                                           
        
        inds1 = np.zeros(params["dim"], dtype=np.int32)
        self.inds = np.zeros(params["dim"] * N1, dtype=np.int32)
        
        for i in range(0, N1):
            if (Ksizes):
                index1DtoND.index(i, inds1, params["SizesK"], params, 1)
            else:
                index1DtoND.index(i, inds1, params["Sizes"], params, 1)

            for j in range(0, params["dim"]):
                if (Ksizes):
                    self.inds[i * params["dim"] + j] =   inds1[j]  if (inds1[j] <= params["KCOs"][j % params["dim"]]) else inds1[j] - params["SizesK"][j % params["dim"]]
                else:
                    self.inds[i * params["dim"] + j] = inds1[j]
                    
        self.inds_gpu = cuda.mem_alloc_like(self.inds)
        cuda.memcpy_htod(self.inds_gpu, self.inds)