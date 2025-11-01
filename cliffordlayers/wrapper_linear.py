# cliffordlayers/wrapper_linear.py

import os
import sys
import ctypes
import numpy as np
from typing import Optional

# Pick the right shared‐lib name
_lib_name = "clifford_linear.dll" if sys.platform == "win32" else "libclifford_linear.so"
# We assume you built it into csrc/build/
_lib_path = os.path.join(os.path.dirname(__file__), "..", "csrc", "build", _lib_name)
_lib = ctypes.CDLL(_lib_path)

# Opaque C struct pointer for your layer object
ClLinearPtr = ctypes.c_void_p

# -- C API declarations --

# CliffordLinear *clifford_linear_create(const int *g_in, int dim, int in_ch, int out_ch, bool use_bias);
_lib.clifford_linear_create.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # g_in
    ctypes.c_int,                  # dim
    ctypes.c_int,                  # in_ch
    ctypes.c_int,                  # out_ch
    ctypes.c_bool,                 # use_bias
]
_lib.clifford_linear_create.restype = ClLinearPtr

# void clifford_linear_forward(const CliffordLinear *L,
#                              const float *x, int B,
#                              float *out);
_lib.clifford_linear_forward.argtypes = [
    ClLinearPtr,                         # layer pointer
    ctypes.POINTER(ctypes.c_float),      # x data
    ctypes.c_int,                        # B (batch size)
    ctypes.POINTER(ctypes.c_float),      # out data
]
_lib.clifford_linear_forward.restype = None

# void clifford_linear_destroy(CliffordLinear *L);
_lib.clifford_linear_destroy.argtypes = [ClLinearPtr]
_lib.clifford_linear_destroy.restype  = None


def c_linear_forward(
    x: np.ndarray,                 # (B, C, D)
    W_mv: np.ndarray,              # (D, O, C)
    b_mv: Optional[np.ndarray],    # (D, O) or None
    g_sig: list                    # signature as list of ints
) -> np.ndarray:
    """
    Run the C implementation of CliffordLinear and return y of shape (B, O, D).
    """
    # 1) shape checks & contiguity
    B, C, D = x.shape
    D2, O, C2 = W_mv.shape
    assert (D2, C2) == (D, C), "W_mv must be (D,O,C)"

    x   = np.ascontiguousarray(x,    dtype=np.float32)
    W_mv= np.ascontiguousarray(W_mv, dtype=np.float32)
    out = np.zeros((B, O, D), dtype=np.float32)

    # 2) create C layer object
    g_arr = (ctypes.c_int * len(g_sig))(*[int(v) for v in g_sig])
    L = _lib.clifford_linear_create(g_arr,
                                     ctypes.c_int(len(g_sig)),
                                     ctypes.c_int(C),
                                     ctypes.c_int(O),
                                     ctypes.c_bool(b_mv is not None))
    if not L:
        raise RuntimeError("clifford_linear_create failed (unsupported dimension?)")

    # 3) cast the pointer to inspect its fields so we can memcpy weight/bias
    class _CStruct(ctypes.Structure):
        _fields_ = [
            ("dim", ctypes.c_int),
            ("n_blades", ctypes.c_int),
            ("g", ctypes.POINTER(ctypes.c_float)),
            ("in_ch", ctypes.c_int),
            ("out_ch", ctypes.c_int),
            ("weight", ctypes.POINTER(ctypes.c_float)),
            ("bias", ctypes.POINTER(ctypes.c_float)),
            ("get_kernel", ctypes.c_void_p),
        ]

    lin = ctypes.cast(L, ctypes.POINTER(_CStruct)).contents

    # 4) copy in the weight array (D*O*C floats)
    nW = D * O * C
    ctypes.memmove(lin.weight,
                   W_mv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   nW * ctypes.sizeof(ctypes.c_float))

    # 5) copy bias if present
    if b_mv is not None:
        b_mv = np.ascontiguousarray(b_mv, dtype=np.float32)
        nB = D * O
        ctypes.memmove(lin.bias,
                       b_mv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       nB * ctypes.sizeof(ctypes.c_float))

    # 6) call forward
    _lib.clifford_linear_forward(
        L,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )

    # 7) destroy and return
    _lib.clifford_linear_destroy(L)
    return out
