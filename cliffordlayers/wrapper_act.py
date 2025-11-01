import os
import sys
import ctypes
import numpy as np

# Load the shared library
_lib_name = "libmv_act.dll" if sys.platform.startswith("win") else "libmv_act.so"
_lib_path = os.path.join(os.path.dirname(__file__), "..", "csrc", "build", _lib_name)
_lib = ctypes.CDLL(_lib_path)

# Pointer types
c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p   = ctypes.POINTER(ctypes.c_int)

# Bind mv_act_forward
_lib.mv_act_forward.argtypes = [
    c_float_p,      # x
    ctypes.c_int,   # B
    ctypes.c_int,   # C
    ctypes.c_int,   # I_in
    ctypes.c_int,   # I_full
    c_int_p,        # input_blades
    c_float_p,      # weight
    ctypes.c_int,   # K
    c_int_p,        # kernel_blades
    c_float_p,      # bias
    c_float_p       # out
]
_lib.mv_act_forward.restype = None

def act_forward(x, weight, input_blades, kernel_blades, I_full, bias):
    """
    Full MV-activation via C: embed → depthwise corr+sigmoid → get.

    Args:
      x             : np.ndarray float32, shape (B, C, I_in)
      weight        : np.ndarray float32, shape (C, K)
      input_blades  : sequence of int, length I_in
      kernel_blades : sequence of int, length K
      I_full        : int, full algebra blade dimension
      bias          : np.ndarray float32, shape (C,)
    Returns:
      out : np.ndarray float32, shape (B, C, I_in)
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    input_blades = np.asarray(input_blades, dtype=np.int32)
    kernel_blades = np.asarray(kernel_blades, dtype=np.int32)
    bias = np.asarray(bias, dtype=np.float32)

    assert x.ndim == 3, f"x must be (B,C,I_in), got {x.shape}"
    B, C, I_in = x.shape
    K = weight.shape[1]
    assert weight.shape == (C, K)
    assert input_blades.shape[0] == I_in
    assert kernel_blades.shape[0] == K
    assert bias.shape == (C,)

    out = np.empty((B, C, I_in), dtype=np.float32)

    _lib.mv_act_forward(
        x.ctypes.data_as(c_float_p),
        ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(I_in),
        ctypes.c_int(I_full),
        input_blades.ctypes.data_as(c_int_p),
        weight.ctypes.data_as(c_float_p),
        ctypes.c_int(K),
        kernel_blades.ctypes.data_as(c_int_p),
        bias.ctypes.data_as(c_float_p),
        out.ctypes.data_as(c_float_p),
    )
    return out
