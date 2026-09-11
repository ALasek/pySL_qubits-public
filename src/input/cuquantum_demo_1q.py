import cupy as cp
import ctypes
from cuquantum import custatevec, cudaDataType, ComputeType

# Create a 1-qubit wavefunction |0?
psi = cp.array([1, 0], dtype=cp.complex64)

# Hadamard matrix
H = cp.array([[1, 1],
              [1, -1]], dtype=cp.complex64) / cp.sqrt(2)

# Create cuStateVec handle
handle = custatevec.create()

# Number of qubits
n_qubits = 1

# Target qubit as ctypes array
target_qubits = (0,)
target_qubits_array = (ctypes.c_int * len(target_qubits))(*target_qubits)

# Get raw pointers
psi_ptr = int(psi.data.ptr)
H_ptr = int(H.data.ptr)

# Use default CUDA stream
stream = cp.cuda.Stream.null.ptr
stream_ptr = int(stream)

# Apply matrix
custatevec.apply_matrix(
    handle,
    psi_ptr,                      # wavefunction pointer
    cp.complex64,                 # data type
    n_qubits,
    H_ptr,                        # matrix pointer
    cp.complex64,                 # data type
    custatevec.MatrixLayout.ROW, # matrix layout
    target_qubits_array,          # target qubits (ctypes array)
    len(target_qubits),
    None,                         # control qubits
    0,                            # num control qubits
    0,                            # control mask
    False,                        # adjoint
    0,                            # workspace size
    None,                         # workspace pointer
    stream_ptr                    # CUDA stream
)

# Destroy handle
custatevec.destroy(handle)

# Print result
print("Updated wavefunction:", psi)
