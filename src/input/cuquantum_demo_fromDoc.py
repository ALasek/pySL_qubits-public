import numpy as np
import cupy as cp
from cuquantum import custatevec
from cuquantum import cudaDataType
from cuquantum import ComputeType


nIndexBits = 3
nSvSize = (1 << nIndexBits)
nTargets = 1
nControls = 2
adjoint = 0

targets = (2,)
controls = (0, 1)

d_sv = cp.asarray([[0.0, 0.0], [0.0, 0.1], [0.1, 0.1], [0.1, 0.2],
                   [0.2, 0.2], [0.3, 0.3], [0.3, 0.4], [0.4, 0.5]], dtype=np.float64)
d_sv = d_sv.view(np.complex128).reshape(-1)

d_sv_result = cp.asarray([[0.0, 0.0], [0.0, 0.1], [0.1, 0.1], [0.4, 0.5],
                          [0.2, 0.2], [0.3, 0.3], [0.3, 0.4], [0.1, 0.2]], dtype=np.float64)
d_sv_result = d_sv_result.view(np.complex128).reshape(-1)

d_matrix = cp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
d_matrix = d_matrix.view(np.complex128).reshape(-1)

# cuStateVec handle initialization
handle = custatevec.create()

# check the size of external workspace
extraWorkspaceSizeInBytes = custatevec.apply_matrix_get_workspace_size(
    handle, cudaDataType.CUDA_C_64F, nIndexBits, d_matrix.data.ptr, cudaDataType.CUDA_C_64F,
    custatevec.MatrixLayout.ROW, adjoint, nTargets, nControls, ComputeType.COMPUTE_64F)

# allocate external workspace if necessary
if extraWorkspaceSizeInBytes > 0:
    workspace = cp.cuda.alloc(extraWorkspaceSizeInBytes)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# apply gate
custatevec.apply_matrix(
    handle, d_sv.data.ptr, cudaDataType.CUDA_C_64F, nIndexBits,
    d_matrix.data.ptr, cudaDataType.CUDA_C_64F, custatevec.MatrixLayout.ROW, adjoint,
    targets, len(targets), controls, 0, len(controls), ComputeType.COMPUTE_64F,
    workspace_ptr, extraWorkspaceSizeInBytes)

# destroy handle
custatevec.destroy(handle)

# --------------------------------------------------------------------------

# check if d_sv holds the updated statevector
correct = cp.allclose(d_sv, d_sv_result)
if not correct:
    raise RuntimeError("example FAILED: wrong result")

# if this is a standalone script, everything is cleaned up properly at exit
