################################################################################################################################
# Real space eigensolver using the CPU
################################################################################################################################

import numpy as np
from numpy import linalg as LA
from src.export.exportEigenvalues import exportEigenvalues
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D

def rSpaceSolverCPU(params, PsiOut, V, thr): #Real space solver GPU
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
    D, eV = LA.eigh(H)

    for i in range(0,5):
        print(D[i])
	


	# Go over eigenstates to construct the init wave function as specified by Input Params
    for J in range(0,5):
        if ((params["InputStates"][2 * J] != 0.0) or (params["InputStates"][2 * J + 1] != 0.0)):
            if (params["TDSESolver"] == "kSpace"):
                PsiOut.wave += (params["InputStates"][2 * J] + 1j * params["InputStates"][2 * J + 1]) * np.reshape(eV[:,J], PsiOut.size)
            else:
                PsiOut.wave += (params["InputStates"][2 * J] + 1j * params["InputStates"][2 * J + 1]) *  eV[:,J]


	
					
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