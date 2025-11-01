# tests/test_linear_c_vs_py.py

import numpy as np
import torch
import pytest

from cliffordlayers.cliffordalgebra import CliffordAlgebra
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear
from cliffordlayers.wrapper_linear import c_linear_forward

#signatures for g
_SIGNATURES = [
    [1.0],
    [-1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0, 1.0],
]

# B ∈ {1,2,3,4}, C ∈ {1,2}, O ∈ {1,2} 
_BC_O_CASES = [
    (b, c, o)
    for b in (1, 2, 3, 4)
    for c in (1, 2)
    for o in (1, 2)
]

@pytest.mark.parametrize("g", _SIGNATURES)
@pytest.mark.parametrize("B,C,O", _BC_O_CASES)
def test_c_matches_python(g, B, C, O):
    # build Python layer
    alg = CliffordAlgebra(g)
    D = alg.n_blades

    x = torch.randn(B, C, D)
    lin = CliffordLinear(g, in_channels=C, out_channels=O, bias=True)
    # randomize weights & bias
    lin.weight.data.uniform_(-2, 2)
    lin.bias  .data.uniform_(-2, 2)

    # run Python
    y_py = lin(x).detach().cpu().numpy()

    # extract raw multivectors + signature
    W_mv = lin.weight.detach().cpu().numpy()  # (D, O, C)
    b_mv = lin.bias  .detach().cpu().numpy()  # (D, O)
    g_sig = g

    # run C
    y_c = c_linear_forward(
        x.detach().cpu().numpy().astype(np.float32),
        W_mv.astype(np.float32),
        b_mv.astype(np.float32),
        g_sig
    )

    # compare
    np.testing.assert_allclose(
        y_c, y_py,
        rtol=1e-5, atol=1e-6,
        err_msg=f"Mismatch for g={g}, B={B}, C={C}, O={O}"
    )
