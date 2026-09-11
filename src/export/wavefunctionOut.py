import numpy as np
import math
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D
import reikna.fft as rfft
import pycuda.gpuarray as gpuarray
from src.get_ccbin import get_ccbin
from src.fft.fftAuxPsi import ToReal

class wavefunctionOut:
    
   
    def exportFull(self, params, step, Filename="wave"):
        n_zero = math.log10((float)(params["TimeSteps"] / params["PrintStep"]) + 1) + 1
        count = int(step / params["PrintStep"])
        index = str(count).zfill((int(n_zero)))
        
        if (step >= 0):
            OutputFileName = "./out/" + params["SimName"] + "/waves/wave_iteration_" + index + ".txt"
        else:
            OutputFileName = "./out/" + params["SimName"] + "/" + Filename + ".txt"
	
        with open(OutputFileName, 'w+') as OutputFile:
            for x in np.nditer(self.wave):
                OutputFile.write("{:e} {:e} \n".format(x.real, x.imag))     
            
    def norm(self):
        norm = sum(self.wave.real**2 + self.wave.imag**2) *self.dV
        return norm
    
    def exportWaveKtoReal(self,  PsiK, thr,  params):

        dim = params["dim"]
        Np = params["Np"]
        Nx = params["Nx"]
        Ny = params["Ny"]
        Nz = params["Nz"]
        
        

        NxLim = int((Nx - 1) / 2)
        NyLim = int((Ny - 1) / 2)
        NzLim = int((Nz - 1) / 2)

        for i in range(0,  params["NOut"]):
            self.wave.flat[i] = 0.0
		

        NLims = np.zeros(dim)
        
        ik = 0
        if (Nx > 1):
            NLims[ik] = NxLim
            ik+=1
        if (Ny > 1):
            NLims[ik] = NyLim
            ik+=1
        if (Nz > 1):
            NLims[ik] = NzLim

        indsR = np.zeros(dim*Np)
        indsK = np.zeros(dim*Np)
	

        for i in range(0, params["N"]):
            index1DtoND.index(i, indsK, params["Sizes"], params)
		# Fill in the padded wave funtion
            for j in range(0, dim*Np):
               indsR[j] =  indsK[j] if indsK[j] <= NLims[j % dim] else indsK[j] - params["Sizes"][j % dim] + self.SizesOutK[j % dim] #indsR[j] = indsK[j] <= NLims[j % params["dim] ? indsK[j] : indsK[j] - params["Sizes[j % params["dim] + SizesOut[j % params["dim];
		
            indNP = indexNDto1D.index(indsR, self.SizesOutK, params)
		#Divide by elements here?
            self.wave.flat[int(indNP)] += PsiK.wave.flat[i]
		

    	#Do the FFT
        #FFTAuxPsi(params, PsiReal, planConvND, false, false, true)
        #self.wave = np.reshape(self.wave, self.size)
        self.wave_gpu.set(self.wave)
        
        self.fftc(self.wave_gpu, self.wave_gpu, 1)
        thr.synchronize()
        self.wave_gpu.get(self.wave)
        ToReal(params, self.wave, True)
        #self.wave=self.wave.ravel()
        
    def __init__(self, params, thr):
        self.NxOut = params["NxOutK"]	#Padded real wave function size
        self.NyOut = params["NyOutK"]
        self.NzOut = params["NzOutK"]
        #self.SizesOutK = params["SizesOutK"]
        self.SizesOutK=[]
        self.size=[]
        for i in range(0,params["dim"]):
            self.SizesOutK.append(int( params["SizesOutK"][i]))
        for i in range(0,params["Np"]):
            self.size += self.SizesOutK
            
        self.wave = np.zeros(self.size, dtype = np.complex64)
        self.wave_gpu=gpuarray.to_gpu(self.wave)
        #self.wave = self.wave.ravel()
            
        fft_h = rfft.FFT(self.wave_gpu)
        self.fftc=fft_h.compile(thr, compiler_options = ['-ccbin', get_ccbin()])
            
        #self.exportWaveKtoReal(PsiK, thr, params)