# cliffordlayers/wrappers_groupnorm.py

import os
import sys
import ctypes
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union

# Pick the right shared‐lib name
_lib_name = "clifford_groupnorm.dll" if sys.platform == "win32" else "libclifford_groupnorm.so"
# We assume you built it into csrc/build/
_lib_path = os.path.join(os.path.dirname(__file__), "..", "csrc", "build", _lib_name)
_lib = ctypes.CDLL(_lib_path)

def c_clifford_group_norm(
    x: torch.Tensor,            # (B, C, *D, I)
    n_blades: int,              # = I
    num_groups: int,  
    running_mean: Optional[torch.Tensor] = None,
    running_cov: Optional[torch.Tensor] = None,
    weight: Optional[Union[torch.Tensor, nn.Parameter]] = None,
    bias: Optional[Union[torch.Tensor, nn.Parameter]] = None,      
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,    
) -> torch.Tensor:
    
    #assert that arguments are consistent
    assert (running_mean is None and running_cov is None) or (running_mean is not None and running_cov is not None)
    assert (weight is None and bias is None) or (weight is not None and bias is not None)

    #tensor -> np.array
    x = x.detach().cpu().numpy().astype(np.float32)

    #shape checks
    B, C, *D, I = x.shape
    assert n_blades == I, "n_blades not equal to I"
    assert C % num_groups == 0, "num_groups must divide C"

    #*D can just be treated as a single flattened dimension
    D_flat = 1
    for dim in D:
        D_flat *= dim

    #setup in and out put
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros((B,C,*D,I), dtype=np.float32)

    #running setup
    if(running_mean is not None):
        assert running_mean.shape == (I, int(C / num_groups)), "running_mean does not have shape: (I, int(C / num_groups))"
        assert running_cov.shape == (I, I, int(C / num_groups)), "running_cov deos not have shape: (I, I, int(C / num_groups))"
        running_mean = np.ascontiguousarray(running_mean, dtype=np.float32)
        running_cov = np.ascontiguousarray(running_cov, dtype=np.float32)
        running_mean_ptr = running_mean.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        running_cov_ptr = running_cov.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        running_mean_ptr = ctypes.c_void_p(0)
        running_cov_ptr = ctypes.c_void_p(0)

    #running setup
    if(weight is not None):
        assert weight.shape == (I, I, int(C / num_groups)), "weight does not have shape: (I, I, int(C / num_groups))"
        assert bias.shape == (I, int(C / num_groups)), "bias does not have shape: (I, int(C / num_groups))"
        weight = np.ascontiguousarray(weight, dtype=np.float32)
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        weight_ptr = weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        weight_ptr = ctypes.c_void_p(0)
        bias_ptr = ctypes.c_void_p(0)
    
    # call into C
    _lib.clifford_groupnorm(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
        ctypes.c_int(C),
        ctypes.c_int(D_flat),
        ctypes.c_int(I),
        ctypes.c_int(num_groups),
        ctypes.c_bool(running_mean is not None),
        running_mean_ptr,
        running_cov_ptr,
        ctypes.c_bool(weight is not None),
        weight_ptr,
        bias_ptr,
        ctypes.c_bool(training),
        ctypes.c_float(momentum),
        ctypes.c_float(eps),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    return torch.from_numpy(out) #return a tensor cast (wrapper provides exactly the same interface)
