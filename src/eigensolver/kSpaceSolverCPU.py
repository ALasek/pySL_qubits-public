################################################################################################################################
# Real space eigensolver using the CPU
################################################################################################################################

import numpy as np
from numpy import linalg as LA
from src.export.exportEigenvalues import exportEigenvalues
from src.input.potentialKSpace import potentialKSpace
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D
from src.get_ccbin import get_ccbin
import reikna.fft as rfft
import pycuda.gpuarray as gpuarray
from src.fft.fftAuxPsi import ToReal

def kSpaceSolverCPU(params, PsiOut, mod, thr): #Real space solver GPU
	#read input parameters from params & declare stuff

	

    KxCO = params["KxECO"]
    KyCO = params["KyECO"]
    KzCO = params["KzECO"]

    Kx = 2 * KxCO + 1
    Ky = 2 * KyCO + 1
    Kz = 2 * KzCO + 1
    
    dKx = params["dKx"]
    dKy = params["dKy"]
    dKz = params["dKz"]
    
    Np = params["Np"]
    dim = params["dim"]
    PsiSize = int((Kx * Ky * Kz) ** Np)
	
    
    SizesK = []
    SizesKE = []
    KCOs = []
    dKs = []
    for i in range(0,dim):
        if (i == 0):
            SizesK.append(Kx)
            SizesKE.append(4 * KxCO + 1)
            KCOs.append(KxCO)
            dKs.append(dKx)
        elif (i ==1):
            SizesK.append(Ky)
            SizesKE.append(4 * KyCO + 1)
            KCOs.append(KyCO)
            dKs.append(dKy)
        elif (i ==2):
            SizesK.append(Kz)
            SizesKE.append(4 * KzCO + 1)
            KCOs.append(KzCO)
            dKs.append(dKz)
	
	#generate Hamiltonian matrix for eigenfunction solving
	
	#Init Hamiltonian
    H = np.zeros(PsiSize*PsiSize, dtype = np.complex64)
    
    Vk = potentialKSpace(params, mod, thr, True)
    #Vk = np.zeros(((4 * KxCO + 1) * (4 * KyCO + 1) * (4 * KzCO + 1)))

    indsCol = np.zeros(dim*Np, dtype=np.int32)
    indsRow = np.zeros(dim*Np, dtype=np.int32)
    indsDiff = np.zeros(dim*Np, dtype=np.int32)
    
    for k in range(0, PsiSize**2):# Go over each row

        j = int(k % PsiSize) # column index
        i = int(np.floor(np.float64(k) / np.float64(PsiSize))) #i is row index

        if (j >= i):
            index1DtoND.index(j, indsCol, SizesK, params)
            index1DtoND.index(i, indsRow, SizesK, params)
            #calulate offset from diagonal
            for l in range(0, dim * Np):
                indsDiff[l] = indsCol[l] - indsRow[l] if  (indsCol[l] - indsRow[l]) >= 0 else (4 * KCOs[l % dim] + 1) + (indsCol[l] - indsRow[l]) #((indsCol[l] - indsRow[l]) >= 0 ? (indsCol[l] - indsRow[l]) : (4 * KCOs[l % dim] + 1) + (indsCol[l] - indsRow[l]))
		
        
    		#fill Hamiltonian, making use of hermiticity to only fill half and do conjugate
            H[j + PsiSize * i] = Vk.pot[indexNDto1D.index(indsDiff, SizesKE, params)]
        
        
            if (i == j): #on - diagonal element
                for p in range(0,Np):
                    for d in range(0,dim):
                        H[j + PsiSize * i] +=  (((indsRow[p * dim + d] - KCOs[d]) * dKs[d]) ** 2) / (2 * params["Mass"][p])
            else:
                H[i + PsiSize * j] = H[j + PsiSize * i].conjugate()
			
			
		#on-site potential
        
    H = H.reshape([PsiSize,PsiSize])			
    
    D, eV = LA.eigh(H)

    for i in range(0,5):
        print(D[i])
	
    
    ccbin = get_ccbin()
    #api = cluda.cuda_api()
    #thr = api.Thread.create()
    PsiTemp = np.zeros(PsiOut.size, dtype = np.complex64)
    PsiTemp_gpu = gpuarray.to_gpu(PsiTemp)
    
    fft_h = rfft.FFT(PsiTemp_gpu)
    fftc=fft_h.compile(thr, compiler_options = ['-ccbin', ccbin])
    

	# Go over eigenstates to construct the init wave function as specified by Input Params
    for J in range(0,5):
        if ((params["InputStates"][2 * J] != 0.0) or (params["InputStates"][2 * J + 1] != 0.0)):
            #PsiTemp = PsiTemp.ravel();
            indsR = np.zeros(dim*Np)
            indsK = np.zeros(dim*Np)
            
            for i in range(0, (Kx*Ky*Kz)**Np):
                index1DtoND.index(i, indsK, SizesK, params)
                for j in range (0, dim*Np):
                    indsR[j] = indsK[j] + params["Sizes"][j % dim] - KCOs[j % dim] if indsK[j] < KCOs[j % dim] else indsK[j] - KCOs[j % dim] #indsR[j] = indsK[j] < KCOs[j % params->dim] ? indsK[j] + params->Sizes[j % params->dim] - KCOs[j % params->dim] : indsK[j] - KCOs[j % params->dim];
            
                indNP = indexNDto1D.index(indsR, params["Sizes"], params)
                PsiTemp[int(indNP)] = eV[i,J]
                
            #ToReal(params, PsiTemp, False)
            #PsiTemp.reshape(PsiOut.size)
            PsiTemp_gpu.set(PsiTemp)
            fftc(PsiTemp_gpu, PsiTemp_gpu, 1)
            thr.synchronize()
            PsiTemp_gpu.get(PsiTemp)
            ToReal(params, PsiTemp, False)
            
            
            if (params["TDSESolver"] == "kSpace"):
                PsiOut.wave += (params["InputStates"][2 * J] + 1j * params["InputStates"][2 * J + 1]) * PsiTemp
            else:
                PsiOut.wave += (params["InputStates"][2 * J] + 1j * params["InputStates"][2 * J + 1]) * PsiTemp.ravel() 

	
					
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


	
    exportEigenvalues(params, D, 5)