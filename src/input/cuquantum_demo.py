import cupy as cp
import ctypes
from cuquantum import custatevec,cudaDataType, ComputeType

# Create wavefunction and matrix
psi = cp.array([1, 0, 0, 0], dtype=cp.complex64)
H = cp.array([[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0]], dtype=cp.complex64)

# Create handle
handle = custatevec.create()

# Number of qubits
n_qubits = 2
target_qubits = (0, 1)
target_qubits_array = (ctypes.c_int * len(target_qubits))(*target_qubits)

# Get raw pointers
psi_ptr = int(psi.data.ptr)
H_ptr = int(H.data.ptr)
stream = int(cp.cuda.Stream.null.ptr)

# Apply matrix
custatevec.apply_matrix(
    handle,
    psi_ptr,
    cp.complex64,
    n_qubits,
    H_ptr,
    cp.complex64,
    custatevec.MatrixLayout.ROW,
    target_qubits_array,
    len(target_qubits),
    None,  # control qubits
    0,     # num control qubits
    0,     # control mask
    False, # adjoint
    0,     # workspace size
    None,  # workspace pointer
    stream
)

# Destroy handle
custatevec.destroy(handle)
