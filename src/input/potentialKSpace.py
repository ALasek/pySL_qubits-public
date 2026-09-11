################################################################################################################################
# Create the potential class for storing the time-independent FFT transformed potential of the problem for k-space, including 2-particle interactions like Coulomb
################################################################################################################################

import numpy as np
import math
import pycuda.driver as cuda
import src.index.index6Dto1D as index6Dto1D
import src.index.index3Dto1D as index3Dto1D
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D
from src.input.electrostaticPotential import electrostaticPotential
from src.input.coulombPotential import CoulombPotential
import reikna.fft as rfft
import pycuda.gpuarray as gpuarray
from src.get_ccbin import get_ccbin

class potentialKSpace():
    
    #copy data from device to host
    def dtoh(self):
        cuda.memcpy_dtoh(self.pot, self.pot_gpu)
    
    #copy data from host to device
    def htod(self):
        cuda.memcpy_htod(self.pot_gpu, self.pot)
        
    #read potential from file specified in params
    def readFromFile(self,params):
        file = open(".\src\\" + params["PotentialFileName"])
        self.filePot = np.zeros(params["N1"])
        lines = file.readlines()
        for i in range(0,params["N1"]):
            self.filePot[i] = float(lines[i+3].split()[0])
        
    # Build the potential from analytically
    def construct(self,params, mod, thr):
        CoulombPotential = mod.get_function("CoulombPotential")
        ccbin = get_ccbin()
        #self.api = cluda.cuda_api()
        #self.thr = api.Thread.create()
        
        Xa = params["Xa"]
        Xb = params["Xb"]
        Ya = params["Ya"]
        Yb = params["Yb"]
        Za = params["Za"]
        Zb = params["Zb"]
        Nx = params["Nx"]
        Ny = params["Ny"]
        Nz = params["Nz"]


        Np = int(params["Np"])
        dim = int(params["dim"])


        x = np.linspace(Xa, Xb, Nx)
        y = np.linspace(Ya, Yb, Ny)
        z = np.linspace(Za, Zb, Nz)
	
        if (params["ImportPotential"]):
            self.readFromFile(params)


        inds = np.zeros(dim*Np)

        if (Np > 1):
            interactionsNr = int((Np**2 - Np) / 2) #Nr of onteractions between particle pairs
            interactionsMap = np.zeros([interactionsNr,2]) #map of which particle interacts with which
		


            for i in range(0,Np):
                for j in range(i+1,Np):
                    interactionsMap[i,0] = i
                    interactionsMap[i,1] = j
        else:
            interactionsNr = 0

        if (interactionsNr > 0):
            V2P=np.zeros((Nx * Ny * Nz)**2,  dtype = np.float32) #store 2-particle potential

            V2P_gpu = cuda.mem_alloc_like(V2P)
            
            blockSize = 256 
            numBlocks = math.ceil(params["N"] / blockSize) 
            #calculate 2-particle potential using Coulomb ionteraction
            CoulombPotential(np.int32(params["N"]), V2P_gpu,  np.int32(Nx),  np.int32(Ny),  np.int32(Nz),  np.float32(Xa),   np.float32(params["dx"]),
                                  np.float32(Ya),   np.float32(params["dy"]),   np.float32(Za),   np.float32(params["dy"]),   np.float32(params["RelPerm"]),
                                   np.float32(params["sigmaFit"][0]),    np.float32(params["sigmaFit"][1]),    np.float32(params["sigmaFit"][2]),
                                    np.float32(params["sigmaFit"][3]),    np.float32(params["sigmaFit"][4]),
                                   np.float32(params["sigmaFit"][5]),    np.float32(params["sigmaFit"][6]),   np.float32(params["sigmaFit"][7]),    np.float32(params["sigmaFit"][8]),
                                  block=(blockSize,1,1), grid=(numBlocks,1))

            cuda.memcpy_dtoh(V2P, V2P_gpu)
            V2P = V2P.astype(np.complex64)

            #Do FFT
        
            V2P=V2P.reshape(self.sizesNP)
            V2P_gpu=gpuarray.to_gpu(V2P)
            
            fft_h2P = rfft.FFT(V2P_gpu)
            self.fftc2P=fft_h2P.compile(thr, compiler_options = ['-ccbin', ccbin])
            
            self.fftc2P(V2P_gpu, V2P_gpu, 0)
            thr.synchronize()
            V2P_gpu.get(V2P)
            V2P=V2P.ravel()
    		#for each pair, fill the potential with their corresponding 2-particle interaction potential
            for p in range(0,interactionsNr):

                p1 = interactionsMap[p,0]
                p2 = interactionsMap[p,1]
                inds2P = np.zeros(dim*2)
                
                for i in range(0, (self.Kx*self.Ky*self.Kz)**2):
                    index1DtoND.index(i, inds2P, self.sizesK, params, 2)
                    inds = np.zeros(dim*Np)
                    for j in range(0,params["dim"]):
                        inds[j + int(dim*p1)] = inds2P[j]
                        inds[j + int(dim*p2)] = inds2P[j + dim]
                        inds2P[j] = inds2P[j] if (inds2P[j] <= self.kLims[j]) else inds2P[j] + params["Sizes"][j] - self.sizesK[j] #inds2P[j] <= KLims[j] ? inds2P[j] : inds2P[j] + params->Sizes[j] - SizesK[j];
                        inds2P[j + dim] = inds2P[j + dim] if (inds2P[j + dim] <= self.kLims[j]) else inds2P[j + dim] + params["Sizes"][j] - self.sizesK[j]#inds2P[j + params->dim] <= KLims[j] ? inds2P[j + params->dim] : inds2P[j + params->dim] + params->Sizes[j] - SizesK[j];
				
                    indND = int(indexNDto1D.index(inds, self.sizesK, params))
                    ind2D = int(indexNDto1D.index(inds2P, params["Sizes"], params, 2))
                    self.pot[indND] += params["Charge"][int(p1)] * params["Charge"][int(p2)] * V2P[ind2D] / (Nx*Ny*Nz*Nx*Ny*Nz)#NORMALIZE HERE?



    	#Add 1 particle static potential
        V1P = np.zeros(Nx * Ny * Nz, dtype = np.complex64)
        
        for p in range(0, params["Np"]):

            for i in range(0,Nx):
                for j in range(0,Ny):
                    for k in range(0,Nz):
                        VVal = 0.0;
                        ind = index3Dto1D.index(i, j, k, Nx, Ny, Nz)
                        
                        #either import from file, or calculate from electrostatic equations
                        if (params["ImportPotential"]):
                            VVal= self.filePot[ind]
                        else:
                            VVal = electrostaticPotential(params,p,x[i],y[j],z[k])

                        if (VVal > 100):
                            VVal = 100.0
					

                        V1P[ind] = VVal
		
            #DO FFT
            V1P=V1P.reshape(self.sizes)
            V1P_gpu=gpuarray.to_gpu(V1P)
            
            fft_h1P = rfft.FFT(V1P_gpu)
            self.fftc1P=fft_h1P.compile(thr, compiler_options = ['-ccbin', ccbin])
            
            self.fftc1P(V1P_gpu, V1P_gpu, 0)
            thr.synchronize()
            V1P_gpu.get(V1P)
            V1P=V1P.ravel()
            
            inds1P = np.zeros(dim)
            #go into if right indexes are 0, i.e. do this other way around ...
            for i in range(0,(self.Kx*self.Ky*self.Kz)):
                index1DtoND.index(i, inds1P, self.sizesK, params, 1)
                inds = np.zeros(dim*Np)
            
                for j in range(0, params["dim"]):
                    inds[j + dim*p] = inds1P[j]
                    inds1P[j] = inds1P[j]  if (inds1P[j] <= self.kLims[j]) else inds1P[j] + params["Sizes"][j] - self.sizesK[j] #inds1P[j] <= KLims[j] ? inds1P[j] : inds1P[j] + params->Sizes[j] - SizesK[j];
			
                indND = indexNDto1D.index(inds, self.sizesK, params)
                ind1D = indexNDto1D.index(inds1P, params["Sizes"], params, 1)
                self.pot[int(indND)] += V1P[int(ind1D)] / (Nx*Ny*Nz) #NORMALIZE?


            
    
    def __init__(self,params, mod, thr, enlarged = False):        
        
        KxCO = params["KxECO"]
        KyCO = params["KyECO"]
        KzCO = params["KzECO"]


	#Check if enlarged, i.e. 2x bigger for writing into esolver Hamiltonian
        if (enlarged):
            KxLim = 2 * KxCO 
            KyLim = 2 * KyCO 
            KzLim = 2 * KzCO
        else:
            KxLim = KxCO
            KyLim = KyCO
            KzLim = KzCO
	

        self.Kx = 2 * KxLim + 1
        self.Ky = 2 * KyLim + 1
        self.Kz = 2 * KzLim + 1
    
        self.pot=np.zeros((self.Kx*self.Ky*self.Kz)**int(params["Np"]), dtype=np.complex64)
        
        self.pot_gpu=cuda.mem_alloc(self.pot.nbytes)
        
        self.sizes = []
        self.sizesK = []
        self.kLims = []
        self.sizesNP=[]
        for i in range(0,params["dim"]):
            if (i == 0):
                self.sizes.append(params["Nx"])
                self.sizesK.append(self.Kx)
                self.kLims.append(KxLim)
            elif (i ==1):
                self.sizes.append(params["Ny"])
                self.sizesK.append(self.Ky)
                self.kLims.append(KyLim)
            elif (i ==2):
                self.sizes.append(params["Ny"])
                self.sizesK.append(self.Kz)
                self.kLims.append(KzLim)
                
        for p in range(0,params["Np"]):
            self.sizesNP += self.sizes
            
        self.construct(params, mod, thr)

        self.htod()