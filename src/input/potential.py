################################################################################################################################
# Create the potential class for storing the time-independent potential of the problem, including 2-particle interactions like Coulomb
################################################################################################################################

import math
import numpy as np
import pycuda.driver as cuda
import src.index.index6Dto1D as index6Dto1D
import src.index.index3Dto1D as index3Dto1D
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D
from src.input.electrostaticPotential import electrostaticPotential
#from src.input.coulombPotential import CoulombPotential

class potential():
    
    #copy data from device to host
    def dtoh(self):
        cuda.memcpy_dtoh(self.pot, self.pot_gpu)
    
    #copy data from host to device
    def htod(self):
        cuda.memcpy_htod(self.pot_gpu, self.pot)
        
    #read potential from file specified in params
    def readFromFile(self,params):
        file = open("./input/" + params["PotentialFileName"])
        self.filePot = np.zeros(params["N1"])
        lines = file.readlines()
        for i in range(0,params["N1"]):
            self.filePot[i] = float(lines[i+3].split()[0])
        
    # Build the potential from analytically
    def construct(self,params, mod):
        CoulombPotential = mod.get_function("CoulombPotential")
        
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
            V2P=np.zeros((Nx * Ny * Nz)**2, dtype = np.float32) #store 2-particle potential
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
            

    		#for each pair, fill the potential with their corresponding 2-particle interaction potential
            for p in range(0,interactionsNr):

                p1 = interactionsMap[p,0]
                p2 = interactionsMap[p,1]
                inds2P = np.zeros(dim*2)

                for i in range(0,params["N"]):
                    index1DtoND.index(i, inds, params["Sizes"], params)

                    for j in range(0,dim):
                        inds2P[j] = inds[int(p1 * dim + j)]
                        inds2P[j + dim] = inds[int(p2 * dim + j)]
				
                    ind2P = int(indexNDto1D.index(inds2P, params["Sizes"], params, 2))
                    self.pot[i] += params["Charge"][int(p1)] * params["Charge"][int(p2)] * V2P[ind2P]



	    #Add 1 particle static potential
        V1P = np.zeros(Nx * Ny * Nz)
        
        for p in range(0, params["Np"]):

            for i in range(0,Nx):
                for j in range(0,Ny):
                    for k in range(0,Nz):
                        VVal = 0.0
                        ind = index3Dto1D.index(i, j, k, Nx, Ny, Nz)
                        
                        #either import from file, or calculate from electrostatic equations
                        if (params["ImportPotential"]):
                            VVal= self.filePot[ind]
                        else:
                            VVal = electrostaticPotential(params,p,x[i],y[j],z[k])

                        if (VVal > 100):
                            VVal = 100.0
					

                        V1P[ind] = VVal
		
        inds1P = np.zeros(dim)
		

		#Only go into if right indexes are 0, i.e. do this other way around ...
        for i in range(0,params["N"]):
            index1DtoND.index(i, inds, params["Sizes"], params)

            for j in range(0, dim):
                inds1P[j] = inds[p * dim + j]
			
            ind1P = indexNDto1D.index(inds1P, params["Sizes"], params, 1)
            self.pot[i] += V1P[int(ind1P)]


            
    
    def __init__(self,params, mod):        
        
        self.pot=np.zeros(params["N"], dtype=np.float32)
        
        self.pot_gpu=cuda.mem_alloc(self.pot.nbytes)
    
        self.construct(params, mod)
        
        self.size1P = []
        self.size = []
        for i in range(0,params["dim"]):
            self.size1P.append(int(params["Sizes"][i]))
        for i in range(0,params["Np"]):
            self.size += self.size1P
        
        #self.pot = np.reshape(self.pot, self.size)

        self.htod()