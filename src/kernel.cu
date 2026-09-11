#include <pycuda-complex.hpp>
//#include <math.h>
//#include <complex>
//#include <cmath>

/*******************************************************************************************************************************
Calculate Coulomb potential
*******************************************************************************************************************************/
__global__ void CoulombPotential(int N, float* V2P, int Nx, int Ny, int Nz, float Xa, float dx, float Ya, float dy, float Za, float dz, float RelPerm,
                                 float sigmaFit0, float sigmaFit1, float sigmaFit2, float sigmaFit3, float sigmaFit4, float sigmaFit5, float sigmaFit6, 
                                 float sigmaFit7, float sigmaFit8){       
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int indOut = ind;
    if (ind<N){
    
	long stride = N;
	long temp = 0;
	
	temp = stride;
	stride = (long)((double)temp / (double)Nx);
	int i1 = ind / stride;
	ind = ind - i1 * stride;
	
	temp = stride;
	stride = (long)((double)temp / (double)Ny);
	int j1 = ind / stride;
	ind = ind - j1 * stride;
	
	temp = stride;
	stride = (long)((double)temp / (double)Nz);
	int k1 = ind / stride;
	ind = ind - k1 * stride;
	
	temp = stride;
	stride = (long)((double)temp / (double)Nx);
	int i2 = ind / stride;
	ind = ind - i2 * stride;
	
	temp = stride;
	stride = (long)((double)temp / (double)Ny);
	int j2 = ind / stride;
	ind = ind - j2 * stride;
	
	temp = stride;
	stride = (long)((double)temp / (double)Nz);
	int k2 = ind / stride;
	//ind = ind - k2 * stride;
	
    
    float x1 = Xa + i1*dx;
    float y1 = Ya + j1*dy;
    float z1 = Za + k1*dz;
    float x2 = Xa + i2*dx;
    float y2 = Ya + j2*dx;
    float z2 = Za + k2*dx;
        
     float d = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2) + pow((z1 - z2), 2));
     float dsoft = powf(d, 8) * sigmaFit0 + powf(d, 7) * sigmaFit1 + powf(d, 6) * sigmaFit2 +
									powf(d, 5) * sigmaFit3 + powf(d, 4) * sigmaFit4 + powf(d, 3) * sigmaFit5 +
									powf(d, 2) * sigmaFit6 + d * sigmaFit7 + sigmaFit8;
	 double VVal = (18.8950673 / RelPerm) * powf(dsoft, -1.0); //use Coulomb softening with cap at 100, 18.89... is conversion factor to natural units and nm
     VVal = (18.8950673 / RelPerm) * powf(dsoft, -1.0); //use Coulomb softening with cap at 100, 18.89... is conversion factor to natural units and nm
     if (VVal>100.0){
         VVal=100.0;
    }
     V2P[indOut]=VVal;
     
   }

}


/*******************************************************************************************************************************
Multiply by potential, calculate time dependent potential part
*******************************************************************************************************************************/
__global__ void Axpy(pycuda::complex<float>* Psi, float* V, 
					int N, float t, 
					int* kernelInds, float Nm1, int Np, int N1, int dim, 
					float ASAW, float kSAW, float SAWoffset, float wSAW, float cSAW, 
					float RS, float DRt0, float URt0, float ATB2, float s2TB2, 
					float* Mass, float* dXs, float Xa, float Ya) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {

		float Vt = 0.0; //some potentially time depentent part of V

		// if (ASAW > 0.0) { // add SAW potential if necessary
			for (int p = 0; p < Np; p++) {
				// Set current coordinates
			
				int ind = ((index / (int)powf(N1, Np - p - 1)) % (N1)) * dim;
				float x = Xa + kernelInds[ind] * dXs[0];
				float y = dim > 1 ? Ya + kernelInds[ind + 1] * dXs[1] : 0.0;
				// Add SAW potential
				Vt += ASAW * (1.0 - cosf(kSAW * (x - SAWoffset) - wSAW * t));
				float TBRamp = 0.5 * (tanhf(RS * (y + (t - DRt0) * cSAW)) + tanhf(-RS * (y + (t - URt0) * cSAW)));
				Vt -= ATB2 * Mass[p] * expf(-s2TB2 * powf(x, 2.0) / 2.0) * TBRamp;
			}
		// }

		Psi[index] *= (V[index] + Vt) * Nm1;
		//Psi[index].y *= (V[index] + Vt) * Nm1;
	}
}

/*******************************************************************************************************************************
Trace out dimensions for export
*******************************************************************************************************************************/

__global__ void AddTrace(int n, pycuda::complex<float>* trace, pycuda::complex<float>* Psi, int p1, int d1, int p2, int d2, int N1, int Np, int dim, float diff, int* Sizes, int* kernelInds) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		//int offset = i * params->dim * params->ParticlesNr; // this is an offset to know where to look in kernelInds for given i
		//Index1DtoNDkernel(i, kernelInds, Sizes, params, offset);
		int indp1 = ((i / (int)powf(N1, Np - p1 -2)) % (N1)) * dim;
		int indp2 = ((i / (int)powf(N1, Np - p2 -2)) % (N1)) * dim;
		int indTrace = kernelInds[indp2 + (d2 - 1) ] + Sizes[d2 - 1] * kernelInds[indp1 + (d1 - 1)];
		trace[indTrace] += (powf(Psi[i].real(), 2.0) + powf(Psi[i].imag(), 2.0)) * diff;
	}
}



/*******************************************************************************************************************************
Momentum-space Staggered Leapfrog solver
*******************************************************************************************************************************/
__global__ void kSpaceSL(int N, pycuda::complex<float>* Psi, pycuda::complex<float>* PsiConv, 
						float t, float dt, int Np, int N1K, int dim, float value, int* Sizes, 
						float* Mass, float* dKs, int iterateRe, int* kernelIndsK) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//int stride = blockDim.x * gridDim.x;
	//float reC, imC;


	if (index < N) {
		
		const   pycuda::complex<float> j(0.0,1.0);    
		int indN = 0;

		int l = (int)pow((float)dim, (float)Np);
		for (int i = 0; i < l; i++) {
			int stride = 1;
			for (int j = i + 1; j < l; j++) {
				stride *= Sizes[j % dim];
			}
			int p = (i ) / dim;
			int indp = ((index / (int)powf(N1K, Np - p - 1)) % (N1K)) * dim;
			int ind = kernelIndsK[indp + (i % dim)] < 0 ? kernelIndsK[indp + (i % dim)] + Sizes[i % dim] : kernelIndsK[indp + (i % dim)];
			indN += ind * stride;
		}


		//Staggered leapfrog in K space
		if (iterateRe == 0) {
			Psi[indN] += 2.0 * dt * (PsiConv[indN].imag());
			for (int d = 0; d < dim; d++) {
				for (int p = 0; p < Np; p++) {
					int indK = ((index / (int)powf(N1K, Np - p - 1)) % (N1K)) * dim;
					Psi[indN] += (value / Mass[p]) * powf(kernelIndsK[indK + d] * dKs[d], 2) * Psi[indN].imag();
				}
			}
		}
		else {
			Psi[indN] += pycuda::complex<float>(0.0, -2.0 * dt * (PsiConv[indN].real()));
			for (int d = 0; d < dim; d++) {
				for (int p = 0; p < Np; p++) {
					int indK = ((index / (int)powf(N1K, Np - p - 1)) % (N1K)) * dim;
					Psi[indN] -=  pycuda::complex<float>(0.0, (value / Mass[p]) * powf(kernelIndsK[indK + d] * dKs[d], 2) * Psi[indN].real());
				}
			}
		}
	}
}



/*******************************************************************************************************************************
Real-space Staggered Leapfrog solver
*******************************************************************************************************************************/
__global__ void rSpaceSL(int N, float* PsiRe, float* PsiIm, float* V, bool* domain, 
                        float t, int iterateRe, int* kernelInds, int Np, int N1, int dim, 
                        float ASAW, float kSAW, float SAWoffset, float wSAW, float cSAW, 
                        float RS, float DRt0, float URt0, float ATB2, float s2TB2, 
                        float fbb, float* faa, int* Sizes, float* Mass, float* dXs, float Xa, float Ya) {

	int index = blockIdx.x * blockDim.x + threadIdx.x; // CUDA index

	if (index < N && domain[index]==1) { //check if index is within the wavefunction ant NOT on a boundary

		float Vt = 0.0; //some potentially time depentent part of V

		for (int p = 0; p < Np; p++) {
			// Set current coordinates
			int ind = ((index / (int)powf(N1, Np - p - 1)) % (N1)) * dim;
			float x = Xa + kernelInds[ind] * dXs[0];
			float y = dim > 1 ? Ya + kernelInds[ind + 1] * dXs[1] : 0.0;
			// Add time-dependent SAW potential
			Vt += ASAW * (1.0 - cosf(kSAW * (x - SAWoffset) - wSAW * t));
      // time-dependent ramp potential
			float TBRamp = 0.5 * (tanhf(RS * (y + (t - DRt0) * cSAW)) + tanhf(-RS * (y + (t - URt0) * cSAW)));
			Vt -= ATB2 * Mass[p] * expf(-s2TB2 * powf(x, 2.0) / 2.0) * TBRamp;
		}
		
		
		int idxNearestNeighbor=1;
		if (iterateRe) { // Iterate real part of the wave function
			PsiRe[index] += (V[index] + Vt) * fbb * PsiIm[index]; // The potential part
			for (int d = 0; d < dim; d++) {
				for (int p = 0; p < Np; p++) {
					PsiRe[index] += (faa[d] / Mass[p]) * (2.0 * PsiIm[index]); // On-site kinetic term

					// Calculate the index offset of the nearest neighbor lattice site
					if (d == (dim - 1))
						idxNearestNeighbor = 1;
					else if (d == (dim - 2))
						idxNearestNeighbor = Sizes[dim - 1];
					else if (d == (dim - 3))
						idxNearestNeighbor = Sizes[dim - 1]* Sizes[dim - 2];

					int pp = Np - p -1;
					while (pp > 0) {
						idxNearestNeighbor *= N1;
						pp--;
					}

					PsiRe[index] -= (faa[d] / Mass[p]) * PsiIm[index - idxNearestNeighbor]; 
					PsiRe[index] -= (faa[d] / Mass[p]) * PsiIm[index + idxNearestNeighbor];
					
				}
			}
		}
		else { // Iterate imaginary part of the wave function
			PsiIm[index] -= (V[index] + Vt) * fbb * PsiRe[index];
			for (int d = 0; d < dim; d++) {
				for (int p = 0; p < Np; p++) {
					PsiIm[index] -= (faa[d] / Mass[p]) * (2.0 * PsiRe[index]); // On-site kinetic term
          
					// Calculate the index offset of the nearest neighbor lattice site
					if (d == (dim - 1))
						idxNearestNeighbor = 1;
					else if (d == (dim - 2))
						idxNearestNeighbor = Sizes[dim - 1];
					else if (d == (dim - 3))
						idxNearestNeighbor = Sizes[dim - 1] * Sizes[dim - 2];

					int pp = Np - p -1;
					while (pp > 0) {
						idxNearestNeighbor *= N1;
						pp--;
					}

    				PsiIm[index] += (faa[d] / Mass[p]) * PsiRe[index - idxNearestNeighbor];
					PsiIm[index] += (faa[d] / Mass[p]) * PsiRe[index + idxNearestNeighbor];
				}
			}
		}
	}
}
