################################################################################################################################
# Create the auxillary wavefunction class for use in convolutions with the potential for k-Space leapfrog
################################################################################################################################



#from src.input.wavefunction import wavefunction
import reikna.fft as rfft
import pycuda.gpuarray as gpuarray
from src.get_ccbin import get_ccbin

#import time


class convWavefunction:
   
    def fft(self, thr, inv = 0):
        #Perform fft or ifft
        self.fftc(self.wave_gpu, self.wave_gpu, inv)
        thr.synchronize()
    
    #def convFFT(self, psi, V, mod, kernelInds, t, params,  N, Nm1,  Np,  N1,  dim,  ASAW,  kSAW,  SAWoffset,  wSAW,  cSAW,  RS,  DRt0,  URt0,  ATB2,  s2TB2, Mass,  dXs,  Xa,  Ya):
        #psi.wave_gpu.get(self.wave_gpu)
        #self.fft(1)

        #blockSize = 256
        #numBlocks = math.ceil(N / blockSize)
        
        #Axpy = mod.get_function("Axpy")
        
        #Axpy(self.wave_gpu, V,  N,  t, kernelInds, Nm1,  Np,  N1,  dim,  ASAW,  kSAW,  SAWoffset,  wSAW,  cSAW,  RS,  DRt0,  URt0,  ATB2,  s2TB2, Mass,  dXs,  Xa,  Ya, block=(blockSize,1,1), grid=(numBlocks,1))
        
        #
    
    def __init__(self, psi, thr, params):
         self.dV = psi.dV
         self.wave_gpu=gpuarray.to_gpu(psi.wave)
         #self.api = cluda.cuda_api()
         #thr = self.api.Thread.create()
         fft_h = rfft.FFT(self.wave_gpu)
         self.fftc=fft_h.compile(thr, compiler_options = ['-ccbin', get_ccbin()])
       