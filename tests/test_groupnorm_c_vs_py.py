# tests/test_groupnorm_c_vs_py.py

import numpy as np
import torch
import pytest
import numpy as np


from cliffordlayers.nn.functional.groupnorm import (
    clifford_group_norm,
)
from cliffordlayers.signature import CliffordSignature
from cliffordlayers.wrapper_groupnorm   import c_clifford_group_norm

@pytest.mark.parametrize("g", [
    [1.0], [-1.0], [1.0,1.0], [-1.0,-1.0], [1.0,1.0,1.0]
])
@pytest.mark.parametrize("B,C,D,num_groups", [
    (4,4,3,2), (5,16,1,8), (7,9,5,3), (16, 40, 9, 8)
])
@pytest.mark.parametrize("running", [
    False, True
])
@pytest.mark.parametrize("scaling", [
    False, True
])
def test_groupnorm_c_matches_python(g, B, C, D, num_groups, running, scaling):

    numerical_stability_iterations = 100

    I = CliffordSignature(g).n_blades
    eps = 1e-5
    if(I > 4):
        eps = 1e-1 #for 3D numerically very unstable with low eps pertubation

    atol = 1e-5
    rtol = 1.3e-6
    if(D*num_groups < 12): #emperical cov calculation less stable => larger limits
        atol = 1e-1
        rtol = 1.3e-2

    for i in range(numerical_stability_iterations):
        x = torch.randn(B, C, D, I)

        rmean_py = None
        rmean_c = None
        rcov_py = None
        rcov_c = None
        if running:
            rmean_py = torch.randn(I, int(C / num_groups))
            rmean_c = rmean_py.clone()
            rcov_py = torch.randn(I,I, int(C / num_groups))
            rcov_c = rcov_py.clone()

        weight = None
        bias = None
        if scaling:
            weight = torch.randn(I,I, int(C / num_groups))
            bias = torch.randn(I, int(C / num_groups))

        x_norm = clifford_group_norm(
                x,
                I,
                num_groups=num_groups,
                running_mean=rmean_py,
                running_cov=rcov_py,
                weight=weight,
                bias=bias,
                eps=eps,
            )

        x_norm_c = c_clifford_group_norm(
            x,
            I,
            num_groups=num_groups,
            running_mean=rmean_c,
            running_cov=rcov_c,
            weight=weight,
            bias=bias,
            eps=eps,
        )
        
        torch.testing.assert_close(x_norm, x_norm_c, atol=atol, rtol=rtol)
        #since we put the tollerances higher for some cases we also check that the mean absolute percentage error is below 5%
        epsilon = 1e-8 
        ape = torch.abs((x_norm - x_norm_c) / (x_norm + epsilon))
        mape = torch.mean(ape) * 100
        assert mape < 5.0, 'mean absolute percentage error is above 5%'


        if running:
            torch.testing.assert_close(rmean_py, rmean_c)
            torch.testing.assert_close(rcov_py, rcov_c)
    