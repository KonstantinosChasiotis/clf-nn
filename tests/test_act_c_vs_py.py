import itertools
import numpy as np
import torch
import pytest

from cliffordlayers.cliffordalgebra import CliffordAlgebra
from cliffordlayers.nn.modules.gcan import MultiVectorAct
from cliffordlayers.wrapper_act import act_forward

# Generate 80 cases: batch ∈ {1,2,4}, channels ∈ {1,2,3}, blades combos of length 1..3 from 0..7
_batches  = [1, 2, 4]
_channels = [1, 2, 3]
_blade_ids = range(8)
_cases = []
for B in _batches:
    for C in _channels:
        for length in [1, 2, 3]:
            for blades in itertools.combinations(_blade_ids, length):
                _cases.append((B, C, blades))
_cases = _cases[:80]

@pytest.mark.parametrize("batch, channels, input_blades", _cases)
def test_shape_and_values(batch, channels, input_blades):
    # 1) Setup
    algebra = CliffordAlgebra([0, 1, 1, 1])
    I_in = len(input_blades)

    # 2) Prepare input
    x_torch = torch.randn(batch, channels, I_in)
    x_np     = x_torch.cpu().detach().numpy().astype(np.float32)

    # 3) Find I_full
    v_full = algebra.embed(x_torch, input_blades)  # shape (B,C,I_full)
    I_full = v_full.size(-1)

    # 4) Python reference
    mva_py = MultiVectorAct(
        channels=channels,
        algebra=algebra,
        input_blades=input_blades,
        agg="linear"
    )
    with torch.no_grad():
        weight_np = np.random.randn(channels, I_in).astype(np.float32)
        bias_np   = np.random.randn(channels).astype(np.float32)
        mva_py.conv.weight.copy_(torch.from_numpy(weight_np[:, None, :]))
        mva_py.conv.bias.copy_(torch.from_numpy(bias_np))
    out_py = mva_py(x_torch).cpu().detach().numpy()  # (B, C, I_in)

    # 5) C implementation
    ib_np = np.array(input_blades, dtype=np.int32)
    out_c = act_forward(
        x_np,
        weight_np,
        ib_np,        # input_blades
        ib_np,        # kernel_blades == input_blades
        I_full,
        bias_np       # per-channel bias
    )

    # 6) Compare
    assert out_c.shape == out_py.shape
    np.testing.assert_allclose(
        out_c, out_py,
        rtol=1e-5, atol=1e-5,
        err_msg=f"Mismatch for B={batch},C={channels},blades={input_blades}"
    )


#DETERMINISTIC TEST WHERE SET THE CONV TO 0 and SEE THAT WE CLEARLY ISOLATE THE BIAS PATH 
def test_bias_only_effect():
    batch, channels, I_in = 3, 4, 5
    algebra = CliffordAlgebra([0,1,1,1])
    # random input
    x_torch = torch.randn(batch, channels, I_in)
    x_np     = x_torch.cpu().detach().numpy().astype(np.float32)

    #build Python module with zero weights, random bias
    mva_py = MultiVectorAct(
        channels=channels,
        algebra=algebra,
        input_blades=tuple(range(I_in)),
        agg="linear"
    )
    with torch.no_grad():
        #zero-out all weights
        mva_py.conv.weight.zero_()
        #random bias
        bias = torch.randn(channels)
        mva_py.conv.bias.copy_(bias)

    out_py = mva_py(x_torch).cpu().detach().numpy()    # shape (B,C,I_in)

    # prepare C inputs
    weight_np = np.zeros((channels, I_in), dtype=np.float32)
    bias_np   = bias.cpu().detach().numpy().astype(np.float32)
    inb_np    = np.arange(I_in, dtype=np.int32)

    # run C
    out_c = act_forward(
        x_np,
        weight_np,
        inb_np,       # input_blades = all blades
        inb_np,       # kernel_blades = same
        I_full=x_torch.size(-1),  # here I_full == I_in
        bias=bias_np
    )

    #check gate = sigmoid(bias), so out[b,c,i] = x[b,c,i]*gate[c]
    gate = 1/(1 + np.exp(-bias_np))
    expected = x_np * gate[None,:,None]

    np.testing.assert_allclose(
        out_c, expected,
        rtol=1e-6, atol=1e-6,
        err_msg="Bias‐only pathway failed"
    )

