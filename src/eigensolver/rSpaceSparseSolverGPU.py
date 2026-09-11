################################################################################################################################
# Real space sparse eigensolver using the GPU
################################################################################################################################

import numpy as np
import ctypes
import pycuda.gpuarray as gpuarray
import os
from src.export.exportEigenvalues import exportEigenvalues
import scipy.sparse as sp
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D

def rSpaceSparseSolverGPU(params, PsiOut, V, thr): #Real space solver GPU
	#read input parameters from params & declare stuff

	
    Nx = params["Nx"]
    Ny = params["Ny"]
    Nz = params["Nz"]
     
    Np = params["Np"]
    dim = params["dim"]
    PsiSize = (Nx * Ny * Nz) ** Np
	
	
	#generate Hamiltonian matrix for eigenfunction solving
	
	#Init Hamiltonian
    H = np.zeros([PsiSize**2], dtype=np.float32)
    
    H2 = params["hbar"] ** 2 #hbar squared
    inds = np.zeros(dim*Np)
    
    for k in range(0, PsiSize):# Go over each row
        ind = k + k*PsiSize
        indPMone=0
        index1DtoND.index(k,inds,params["Sizes"],params)

		#fill Hamiltonian, making use of hermiticity to only fill half and do conjugate

		#on-site potential
        H[ind] += V.pot[k] #!!!!!!!!!!!!!!!!!!!!!!!
        for d in range(0,dim):
            for p in range(0,Np):
                H[ind] += H2 / (params["Mass"][p] * params["dXs"][d] ** 2) # on-site kinetic part
                
                inds[p*dim+d]-=1
                indPMone = indexNDto1D.index(inds,params["Sizes"],params)
                if (inds[p*dim+d]>=0 and inds[p*dim+d]< params["Sizes"][d]):
                    H[int(k*PsiSize + indPMone)] -= H2 / (2 * params["Mass"][p] * params["dXs"][d] ** 2)
                    
                inds[p*dim+d]+=2
                indPMone = indexNDto1D.index(inds,params["Sizes"],params)
                
                if (inds[p*dim+d]>=0 and inds[p*dim+d]< params["Sizes"][d] and (k+1)<PsiSize):
                    H[int(k*PsiSize + indPMone)] -=  H2 / (2 * params["Mass"][p] * params["dXs"][d] ** 2)
                    
                inds[p*dim+d]-=1
				
    H=np.reshape(H, [PsiSize,PsiSize])
				

   #Do eig things here
   
    if os.name == 'nt':
       _libcusparse = ctypes.cdll.LoadLibrary('cusparse64_11.dll')
       _libcusolver = ctypes.cdll.LoadLibrary('cusolver64_11.dll')
    else:
        _libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
        _libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')
        
    _libcusparse.cusparseCreate.restype = int
    _libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

    _libcusparse.cusparseDestroy.restype = int
    _libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

    _libcusparse.cusparseCreateMatDescr.restype = int
    _libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]


# cuSOLVER
    

    _libcusolver.cusolverSpCreate.restype = int
    _libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

    _libcusolver.cusolverSpDestroy.restype = int
    _libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]

    _libcusolver.cusolverSpScsreigvsi.restype = int
    _libcusolver.cusolverSpScsreigvsi.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_float,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_float,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]


    Hcsr = sp.csr_matrix(H, dtype=np.float32)
   
    x0 = gpuarray.to_gpu(np.zeros(PsiSize,dtype=np.float32))
    x = gpuarray.to_gpu(np.zeros(PsiSize,dtype=np.float32))
    mu = gpuarray.to_gpu(np.zeros(1,dtype=np.float32))
    
    Hmu =np.zeros(1,dtype=np.float32)
    Hx=np.zeros(PsiSize,dtype=np.float32)

    # Copy arrays to GPU
    dcsrVal = gpuarray.to_gpu(Hcsr.data)
    dcsrColInd = gpuarray.to_gpu(Hcsr.indices)
    dcsrIndPtr = gpuarray.to_gpu(Hcsr.indptr)


    # Create solver parameters
    m = ctypes.c_int(Hcsr.shape[0])  # Need check if A is square
    nnz = ctypes.c_int(Hcsr.nnz)
    descrA = ctypes.c_void_p()
    mu0 = ctypes.c_float(0)
    maxite = ctypes.c_int(1000)
    tol = ctypes.c_float(1e-9)
   
    # create cusparse handle
    _cusp_handle = ctypes.c_void_p()
    status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
    assert(status == 0)
    cusp_handle = _cusp_handle.value

    # create MatDescriptor
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
    assert(status == 0)

    #create cusolver handle
    _cuso_handle = ctypes.c_void_p()
    status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
    assert(status == 0)
    cuso_handle = _cuso_handle.value

    # Solve
    status=_libcusolver.cusolverSpScsreigvsi(cuso_handle,
                                         m,
                                         nnz,
                                         descrA,
                                         int(dcsrVal.gpudata),
                                         int(dcsrIndPtr.gpudata),
                                         int(dcsrColInd.gpudata),
                                         mu0,
                                         int(x0.gpudata),
                                         maxite,
                                         tol,
                                         int(mu.gpudata),
                                         int(x.gpudata))

    
    mu.get(Hmu)
    x.get(Hx)

    print(status)
    

    # Destroy handles
    status = _libcusolver.cusolverSpDestroy(cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(cusp_handle)
    assert(status == 0)

   
    print(Hmu)
	


	# Go over eigenstates to construct the init wave function as specified by Input Params
   
    PsiOut.wave +=  Hx

	
							# do a final normalisation
    if (params["TDSESolver"] == "rSpace"):		# do a final normalisation
        norm = PsiOut.norm()
        PsiOut.wave /= np.sqrt(norm)
    	# Write wave and eigenvalues to file as appropriate
        PsiOut.exportFull( params, -1, "wave")
    elif (params["TDSESolver"] == "kSpace"):
        PsiOut.exportFull( params, -1, "wave")
        PsiOut.wave = np.reshape(PsiOut.wave, PsiOut.size)
        PsiOut.htod()
        PsiOut.fft(thr, 0)
        PsiOut.dtoh()
        norm = PsiOut.norm()
        PsiOut.wave /= np.sqrt(norm)
        PsiOut.exportFull( params, -1, "Kwave")

	
    #ExportEigenvalues(params, D, 5)