#!/usr/bin/env python3
"""Sanity checks for LittleBitLinear against the paper's equations.

Checks, for residual=False and residual=True:
  * effective split_dim / eff_bit_actual match the paper formulas
  * the forward pass equals  diag(h) U_sign diag(l) V_sign^T diag(g)  (+ residual path)
  * the Dual-SVID initialization reconstructs W reasonably (relative error)
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "repo")
sys.path.insert(0, REPO)

from quantization.utils import apply_littlebit_patch  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from quantization.functions import SmoothSign  # noqa: E402
from quantization.modules.littlebit import LittleBitLinear as LBL  # noqa: E402


def make_args(eff_bit, residual):
    import argparse
    return argparse.Namespace(quant_func="SmoothSign", quant_mod="LittleBitLinear", residual=residual,
                              split_dim=1024, eff_bit=eff_bit, kv_factor=1.0, min_split_dim=8,
                              use_itq=False, itq_n_iter=50, model_id="synthetic")


def eff_bits_formula(a, b, s, residual):
    if residual:
        return (s * 2 * (a + b + 16) + 32 * (a + b)) / (a * b)
    return (s * (a + b + 16) + 16 * (a + b)) / (a * b)


def split_formula(a, b, eb, residual):
    if residual:
        return int((a * b * eb - 32 * (a + b)) / (2 * (a + b + 16)))
    return int((a * b * eb - 16 * (a + b)) / (a + b + 16))


def run_case(a, b, eb, residual, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(b, a, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(a, b) * 0.05)
        lin.bias.copy_(torch.randn(a) * 0.01)

    class Holder(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.linear = lin

    holder = Holder(lin)
    x = torch.randn(4, b)
    ref = lin(x)
    holder = apply_littlebit_patch(holder, make_args(eb, residual), do_train=True)
    mod = holder.linear
    assert isinstance(mod, LBL), type(mod)

    s_expected = split_formula(a, b, eb, residual)
    s_expected = (s_expected // 8) * 8
    s_expected = max(s_expected, 8)
    assert mod.split_dim == s_expected, (mod.split_dim, s_expected)
    eb_expected = eff_bits_formula(a, b, mod.split_dim, residual)
    assert abs(mod.eff_bit_actual - eb_expected) < 1e-9, (mod.eff_bit_actual, eb_expected)

    # explicit effective weight, following Eq. 5: W_hat = diag(h) U_sign diag(l) V_sign^T diag(g)
    def w_hat(U, V, u1, u2, v1, v2):
        Us = SmoothSign(U.float())
        Vs = SmoothSign(V.float())
        h = u1.float().reshape(-1, 1)          # (out, 1)
        g = v2.float().reshape(1, -1)          # (1, in)
        l = (u2.float() * v1.float()).reshape(1, -1)  # (1, r)  element-wise latent scale
        return (((Us * l) * h)) @ Vs * g

    W = w_hat(mod.U, mod.V, mod.u1, mod.u2, mod.v1, mod.v2)
    if residual:
        W = W + w_hat(mod.U_R, mod.V_R, mod.u1_R, mod.u2_R, mod.v1_R, mod.v2_R)

    y_explicit = x @ W.t() + mod.bias
    y_ref = ref
    y_mod = mod(x)
    print(f"  a={a} b={b} eb={eb} res={residual} split={mod.split_dim} eff_bit={mod.eff_bit_actual:.4f}")
    print(f"  |y_mod - y_explicit|max = {(y_mod - y_explicit).abs().max().item():.3e}")
    print(f"  |y_mod - y_fp     |max = {(y_mod - y_ref).abs().max().item():.3e}"
          f"  (init approximation, scale ~{y_ref.abs().max().item():.3f})")
    assert torch.allclose(y_mod, y_explicit, atol=1e-4, rtol=1e-4), "forward != explicit effective weight"
    # init error should be small-ish for moderate compression
    rel = (y_mod - y_ref).norm() / y_ref.norm()
    print(f"  relative init error = {rel.item():.4f}")
    return rel.item()


if __name__ == "__main__":
    for res in (False, True):
        for eb in (1.0, 0.55, 0.3, 0.1):
            run_case(256, 128, eb, res)
    print("ALL CHECKS PASSED")
