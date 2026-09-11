import sys
import os
import math
import numpy as np
import pycuda.gpuarray as gpuarray

def exportTrace(Psi, params, step, OutK, kernelInds, mod):
    
    AddTrace = mod.get_function("AddTrace")
    
    Xa = params["Xa"]
    Xb = params["Xb"]
    Ya = params["Ya"]
    Yb = params["Yb"]
    Za = params["Za"]
    Zb = params["Zb"]

    Lx = Xb - Xa  #Length
    Ly = Yb - Ya
    Lz = Zb - Za

    if (OutK):
        N = params["NOut"]
        Nx = params["NxOutK"]
        Ny = params["NyOutK"]
        Nz = params["NzOutK"]
    else:
        N = params["N"]
        Nx = params["Nx"]
        Ny = params["Ny"]
        Nz = params["Nz"]
	


    dx = 1.0
    dy = 1.0
    dz = 1.0

    Sizes = np.zeros(params["dim"], dtype=np.int32)

    ik = 0
    if (Nx > 1):
        Sizes[ik] = Nx
        dx = Lx / (Nx - 1)
        ik=ik+1
    if (Ny > 1):
        Sizes[ik] = Ny
        dy = Ly / (Ny - 1)
        ik=ik+1
    if (Nz > 1):
        Sizes[ik] = Nz
        dz = Lz / (Nz - 1)

	# Memory to store index data for the kernel, as it doesn't like allocating this inside. Could be worked around...
	

	#For each Trace specified by Input Parameters, do the stuff
    for i in range(0, params["TracesNr"]):

        traceid = params["Traces"][i]

		# Get which particle and dimension we plot, i.e. do not trace over
        #p1 = int(traceid // 1000)
        #d1 = int((traceid // 100) % 10)
        #p2 = int(((traceid // 10) % 100) % 10)
        #d2 = int(((traceid % 1000) % 100) % 10)
        p1 = traceid[0]
        d1 = traceid[1]
        p2 = traceid[2]
        d2 = traceid[3]

        if (p1 > params["Np"] or p2 > params["Np"] or d1 > params["dim"] or d2 > params["dim"]):
            print("error: Trace specs in ExportWave wrong")
            sys.exit(1)
            

		
        if (d1 == 1):
            d1s = "X"
            diff1 = dx
        elif (d1 == 2):
            d1s = "Y"
            diff1 = dy
        elif (d1 == 3):
            d1s = "Z"
            diff1 = dz
        else:
            print("error: d1 in ExportWave wrong")
            sys.exit(1)
            

        if (d2 == 1):
            d2s = "X"
            diff2 = dx
        elif (d2 == 2):
            d2s = "Y"
            diff2 = dy
        elif (d2 == 3):
            d2s = "Z"
            diff2 = dz
        else:
            print("error: d2 in ExportWave wrong")
            sys.exit(1)

        FolderName = "Trace" + str(p1) + d1s + str(p2) + d2s

		#allocate Trace
        size = Sizes[d1-1] * Sizes[d2-1]  #size of Trace
        trace = np.zeros(size, dtype = np.complex64)
        trace_gpu = gpuarray.to_gpu(trace)

        diff = ( (dx * dy * dz) ** params["Np"]) / (diff1 * diff2) # size of dV except the dimensions we plot

		# Run the kernel. Doesn't work for some reason
        blockSize = 256
        numBlocks = math.ceil(N / blockSize) 
        
        AddTrace(N, trace_gpu, Psi, p1, d1, p2, d2, params["N1"], params["Np"], params["dim"], diff, Sizes, kernelInds, block=(blockSize,1,1), grid=(numBlocks,1))
		
        trace_gpu.set(trace)

		# Create dir and write each trace to file
        
        
        n_zero = math.log10((float)(params["TimeSteps"] / params["PrintStep"]) + 1) + 1
        count = int(step / params["PrintStep"])
        index = str(count).zfill((int(n_zero)))
        

    
        OutputFileName = "./out/" + params["SimName"] + "/waves/" + FolderName + "wave_iteration_" + index + ".txt"
        #if (step==0):
        #    os.mkdir("./out/" + params["SimName"] + "/waves/" + FolderName)
        with open(OutputFileName, 'w+') as OutputFile:
            for x in np.nditer(trace):
                OutputFile.write("{:e} {:e} \n".format(np.sqrt(x.real), np.sqrt(x.imag)))  
        
      

