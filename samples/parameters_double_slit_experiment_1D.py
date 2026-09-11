import math
import numpy as np
import sys

def getparams():
    
    params = {
        
        "SimName": "Double_slit_experiment_1D",

        ################################################################
        # FLAGS 
        ################################################################
        
        "Np": 1, # Number of particles
        # "IncludeSpin": False, # Solve Pauli equation instead of Schrödinger equation (future release)
        
        "TDSESolver": "rSpace", # Type of staggered leapfrog solver [rSpace, kSpace]
        "Eigensolver": "GPUrSpace", # Type of eigensolver [CPUrSpace, GPUrSpace] future [GPUkSpace, GPUrSparse]

        # Define the initial superposition state of the wave function. 
        # The first two elements are the real and imaginary components of the ground state. 
        # The second two elements are the real and imaginary components of the first excited state. 
        "InputStates": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Import external files (if set to sal)
        "ImportPotential": False, # If set to false, the potential will be created functionally in fftauxV.py
        "PotentialFileName": "potential.txt",

        "ImportWavefunction": True, # If set to false, the wave function will be solved using the specified eigensolver
        "WaveFileName": "wave_DSE1D.txt",
        "NormalizeWaveFunction": True, # Set to false if importing a partial wave function

        "ExportTrace": False, # Export a reduced wave function for speed and smaller disk usage
        "Traces": [ [1,1,2,1] ],

        ################################################################
        # Particle properties
        ################################################################

        # Mass is quoted in [electron masses] (so 1 for a free electron in a vacuum)
        "Mass": np.array([1.0], dtype=np.float32),
        
        # Charge in [electron charges]
        "Charge": np.array([1.0], dtype=np.float32),

        # Relative permittivity of quantum well material
        "RelPerm": 13.1,

        # Conversion between SI units and natural units (hbar = 1, m_e = 1)
        "v_si2nu": 1.312342066e-2,
        "t_si2nu": 1.157676458e2,

        ################################################################
        # Surface acoustic wave
        ################################################################

        # SAW amplitude [meV] , speed of sound [nm]/[ps], wave length [nm], and offset [nm]
        "ASAW": 0.0,
        "cSAW": 3.0,
        "lambdaSAW": 1000.0,
        "SAWoffset": 0.0,

        ################################################################
        # Simulation time
        ################################################################

        
        "dt": 0.0005, # duration of a single time step in [ps]
        "StartTime": 0.0,
        "TimeSteps": 10000,
        "PrintStep": 10, # How often the wave function is saved to a file

        ################################################################
        # Simulation domain
        ################################################################

        # Lattice boundaries [nm]
        "Xa": -200.0,
        "Xb": 200.0,
        "Ya": 0.0,
        "Yb": 0.0,

        "Za": 0.0,
        "Zb": 0.0,

        # Number of lattice sites
        "Nx": 801,
        "Ny": 1,
        "Nz": 1,

        "NxOutK": 61,
        "NyOutK": 61,
        "NzOutK": 1,

        # Momentum Solver Inputs
        "KxECO": 20,
        "KyECO": 20,
        "KzECO": 0,
        "KxKCO": 20,
        "KyKCO": 20,
        "KzKCO": 0,


        # Lattice padding (for expanding TDSE solution window)
        # "padLeft": 0,
        # "padRight": 0,
        # "padTop": 0,
        # "padBottom": 0,

        # Absorbing Boudary Conditions (Leave as zero for Dirichlet boundary)
        # "ABC_L": 0,
        # "ABC_R": 0,
        # "ABC_T": 0,
        # "ABC_B": 0,
        # "ABC_U": 0,
        # "ABC_D": 0,

        ################################################################
        # Potential 
        ################################################################

        # Potential variables
        "w2X": 0.0,
        "w2Y": 0.0,
        "ATB1": 0.0,
        "s2TB1": 0.0,
        "ATB2": 0.0,
        "s2TB2": 0.0,
        "detuneX": 0.0,

        "RS": 0.0,
        "DRt0": 1.0,
        "URt0": 12.4,

        # "sigmaFit": [1.74707037e-17, -2.73761743e-14, 1.78462882e-11, -6.26868417e-09, 1.28479963e-06, -1.55625754e-04, 1.07952353e-02, 6.02006265e-01, 6.93382649e+00],
        # "Traces": 0,
        
    }

    t_si2nu = params["t_si2nu"]
    v_si2nu = params["v_si2nu"]

    # Convert variables to Natural Units
    params["dt"] *= t_si2nu
    params["StartTime"] *= t_si2nu
    params["ASAW"] *= v_si2nu
    params["cSAW"] /= t_si2nu
    params["wSAW"] = 2 * math.pi * params["cSAW"] / params["lambdaSAW"]
    params["kSAW"] = 2 * math.pi  / params["lambdaSAW"]
    params["DRt0"] *= t_si2nu
    params["URt0"] *= t_si2nu

    # Construct Hamiltonian
    params["dx"] = 1.0
    params["dy"] = 1.0
    params["dz"] = 1.0
    if (params["Nx"] > 1):
        params["dx"] = (params["Xb"] - params["Xa"]) / (params["Nx"] - 1)
    if (params["Ny"] > 1):
        params["dy"] = (params["Yb"] - params["Ya"]) / (params["Ny"] - 1)
    if (params["Nz"] > 1):
        params["dz"] = (params["Zb"] - params["Za"]) / (params["Nz"] - 1)
        
        

    if (params["TDSESolver"] != "kSpace"):
        params["NxOutK"] = 1
        params["NyOutK"] = 1
        params["NzOutK"] = 1
        params["KxKCO"] = 0
        params["KyKCO"] = 0
        params["KzKCO"] = 0
	
        
    params["Kx"] = 2 * params["KxKCO"]  + 1
    params["Ky"]  = 2 * params["KyKCO"]  + 1
    params["Kz"] = 2 * params["KzKCO"]  + 1
    
    params["N"] =(params["Nx"] * params["Ny"] * params["Nz"]) ** params["Np"]
    params["N1"] = params["Nx"] * params["Ny"] * params["Nz"]
    params["N1K"] = params["Kx"] * params["Ky"] * params["Kz"]
    params["NOut"] = (params["NxOutK"] * params["NyOutK"] * params["NzOutK"])**params["Np"]
    params["NK"] = ((2 * params["KxKCO"] + 1) * (2 * params["KyKCO"] + 1) * (2 * params["KzKCO"] + 1))**params["Np"]
    
    params["dim"] = 0
    if (params["Nx"] > 1):
        params["dim"]=params["dim"]+1
    if (params["Ny"] > 1):
        params["dim"]=params["dim"]+1
    if (params["Nz"] > 1):
        params["dim"]=params["dim"]+1
    if (params["dim"] == 0):
        print( "No dimension size is greater than 1")
        exit(1)
	

	# Check if padded output is greater than Psi
    if ((params["TDSESolver"] == "kSpace") and (params["NxOutK"] < params["Nx"] or params["NyOutK"] < params["Ny"] or params["NzOutK"] < params["Nz"])) :
        print("NOut parameters cannot be smaller than N for KSpaceLeapfrog")
        sys.exit(1)
	
    if ((params["TDSESolver"] == "kSpace") and ((2 * params["KxKCO"] + 1) >params["Nx"] or (2 * params["KyKCO"] + 1) > params["Ny"] or (2 * params["KzKCO"] + 1) > params["Nz"])):
        print("(2 * Kx/y/zKCO + 1)  cannot be greater than Nx/y/z for KSpaceLeapfrog")
        sys.exit(1)

	
    params["dKx"] =  2 * math.pi / (params["Xb"] - params["Xa"]) if (params["Xb"] - params["Xa"])>0 else 1.0 #delta momentum
    params["dKy"] =  2 * math.pi / (params["Yb"] - params["Ya"]) if (params["Yb"] - params["Ya"])>0 else 1.0
    params["dKz"] =  2 * math.pi / (params["Zb"] - params["Za"]) if (params["Zb"] - params["Za"])>0 else 1.0
    
    params["Sizes"] = np.zeros( params["dim"], dtype = np.int32)
    params["SizesK"] = np.zeros( params["dim"], dtype = np.int32)
    params["SizesOutK"] = np.zeros( params["dim"], dtype = np.int32)
    params["KCOs"] = np.zeros( params["dim"], dtype = np.int32)
    params["dKs"] = np.zeros( params["dim"], dtype = np.float32)
    params["dXs"] = np.zeros( params["dim"], dtype = np.float32)
    params["faa"] = np.zeros( params["dim"], dtype = np.float32)

    params["hbar"] = 1
    
    i = 0
    if (params["Nx"] > 1):
        params["Sizes"][i] = params["Nx"] #Psi size
        params["SizesK"][i] = params["Kx"] #K space size
        params["SizesOutK"][i] = params["NxOutK"]
        params["KCOs"][i] = params["KxKCO"] #K cutoff
        params["dKs"][i] = params["dKx"] #delta K
        params["dXs"][i] = params["dx"]
        params["faa"][i] = params["hbar"] * params["dt"] / (params["dx"] **2) #kinetic parameter faa for Hamiltonians/kernels
        i=i+1
    if (params["Ny"] > 1):
        params["Sizes"][i] = params["Ny"] #Psi size
        params["SizesK"][i] = params["Ky"] #K space size
        params["SizesOutK"][i] = params["NyOutK"]
        params["KCOs"][i] = params["KyKCO"] #K cutoff
        params["dKs"][i] = params["dKy"] #delta K
        params["dXs"][i] = params["dy"]
        params["faa"][i] = params["hbar"] * params["dt"] / (params["dy"] **2) #kinetic parameter faa for Hamiltonians/kernels
        i=i+1
    if (params["Nz"] > 1):
        params["Sizes"][i] = params["Nz"] #Psi size
        params["SizesK"][i] = params["Kz"] #K space size
        params["SizesOutK"][i] = params["NzOutK"]
        params["KCOs"][i] = params["KzKCO"] #K cutoff
        params["dKs"][i] = params["dKz"] #delta K
        params["dXs"][i] = params["dz"]
        params["faa"][i] = params["hbar"] * params["dt"] / (params["dz"] **2) #kinetic parameter faa for Hamiltonians/kernels
        i=i+1
	

    params["fbb"] = 2.0 * params["dt"] / params["hbar"] #time parameter fbb for Hamiltonians/kernels

    return params