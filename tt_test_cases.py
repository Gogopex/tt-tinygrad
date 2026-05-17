"""Test cases and torch references for the TT backend.

`CASES` is a list of (label, thunk) pairs. Each thunk builds and `.realize()`s a
tinygrad expression on the "TT" device, using the module-level `tt_input` to
construct inputs. The runner (`tt_tests.py`) monkey-patches `tt_input` so it
returns real `from_torch(...)` tensors built from deterministic seeds.

`REFS` is a dict of label -> (torch_fn, in_shapes[, slicer]). Each `in_shapes`
entry is either a 2D shape `(r, c)` or `ROW64` = `((64,), (1, 64))` indicating a
1D logical input that's promoted to (1, 64) on device. The V0 in-process runtime
is 2D only, so the two cases that use `ROW64` (matmul_bias, layernorm_affine)
are skipped by the runner.
"""
from __future__ import annotations
import torch
from tinygrad import Tensor

def tt_input(*shape, device="TT", dtype=None):
  return Tensor.empty(*shape, device=device, dtype=dtype)

DIM = 64
ROW64 = ((64,), (1, 64))

CASES = [
  ("add_mul",       lambda: ((tt_input(64, 64, device="TT") + tt_input(64, 64, device="TT")) * 2).realize()),
  ("relu",          lambda: tt_input(64, 64, device="TT").relu().realize()),
  ("mul",           lambda: (tt_input(64, 64, device="TT") * tt_input(64, 64, device="TT")).realize()),
  ("sqrt",          lambda: tt_input(64, 64, device="TT").sqrt().realize()),
  ("sigmoid",       lambda: tt_input(64, 64, device="TT").sigmoid().realize()),
  ("matmul",        lambda: (tt_input(64, 64, device="TT") @ tt_input(64, 64, device="TT")).realize()),
  ("exp",           lambda: tt_input(64, 64, device="TT").exp().realize()),
  ("abs_diff",      lambda: (tt_input(64, 64, device="TT") - tt_input(64, 64, device="TT")).abs().realize()),
  ("matmul_128",    lambda: (tt_input(128, 128, device="TT") @ tt_input(128, 128, device="TT")).realize()),
  ("matmul_rect",   lambda: (tt_input(64, 128, device="TT") @ tt_input(128, 32, device="TT")).realize()),
  ("neg",           lambda: (-tt_input(64, 64, device="TT")).realize()),
  ("recip",         lambda: (tt_input(64, 64, device="TT") + 1).reciprocal().realize()),
  ("log",           lambda: (tt_input(64, 64, device="TT") + 0.1).log().realize()),
  ("tanh",          lambda: tt_input(64, 64, device="TT").tanh().realize()),
  ("matmul_256",    lambda: (tt_input(256, 256, device="TT") @ tt_input(256, 256, device="TT")).realize()),
  ("matmul_relu",   lambda: (tt_input(64, 64, device="TT") @ tt_input(64, 64, device="TT")).relu().realize()),
  ("matmul_add",    lambda: ((tt_input(64, 64, device="TT") @ tt_input(64, 64, device="TT")) + tt_input(64, 64, device="TT")).realize()),
  ("gelu",          lambda: tt_input(64, 64, device="TT").gelu().realize()),
  ("silu",          lambda: tt_input(64, 64, device="TT").silu().realize()),
  ("square",        lambda: tt_input(64, 64, device="TT").square().realize()),
  ("rsqrt",         lambda: (tt_input(64, 64, device="TT") + 0.1).rsqrt().realize()),
  ("maximum",       lambda: tt_input(64, 64, device="TT").maximum(tt_input(64, 64, device="TT")).realize()),
  ("squared_diff",  lambda: (tt_input(64, 64, device="TT") - tt_input(64, 64, device="TT")).square().realize()),
  ("matmul_silu",   lambda: (tt_input(64, 64, device="TT") @ tt_input(64, 64, device="TT")).silu().realize()),
  ("div",           lambda: (tt_input(64, 64, device="TT") / (tt_input(64, 64, device="TT") + 1)).realize()),
  ("sin",           lambda: tt_input(64, 64, device="TT").sin().realize()),
  ("cos",           lambda: tt_input(64, 64, device="TT").cos().realize()),
  ("minimum",       lambda: tt_input(64, 64, device="TT").minimum(tt_input(64, 64, device="TT")).realize()),
  ("scalar_add",    lambda: (tt_input(64, 64, device="TT") + 3.5).realize()),
  ("scalar_mul",    lambda: (tt_input(64, 64, device="TT") * 0.25).realize()),
  ("matmul_long_k", lambda: (tt_input(64, 256, device="TT") @ tt_input(256, 64, device="TT")).realize()),
  ("matmul_thin_m", lambda: (tt_input(32, 128, device="TT") @ tt_input(128, 64, device="TT")).realize()),
  ("sum_axis0",    lambda: tt_input(64, 64, device="TT").sum(axis=0).realize()),
  ("max_axis0",    lambda: tt_input(64, 64, device="TT").max(axis=0).realize()),
  ("sum_128",      lambda: tt_input(128, 128, device="TT").sum(axis=0).realize()),
  ("sum_256_64",   lambda: tt_input(256, 64, device="TT").sum(axis=0).realize()),
  ("sum_64_256",   lambda: tt_input(64, 256, device="TT").sum(axis=0).realize()),
  ("sum_axis1",    lambda: tt_input(64, 64, device="TT").sum(axis=1).realize()),
  ("max_axis1",    lambda: tt_input(64, 64, device="TT").max(axis=1).realize()),
  ("sum_ax1_long", lambda: tt_input(64, 256, device="TT").sum(axis=1).realize()),
  ("mean_axis0",   lambda: tt_input(64, 64, device="TT").mean(axis=0).realize()),
  ("mean_axis1",   lambda: tt_input(64, 64, device="TT").mean(axis=1).realize()),
  ("min_axis0",    lambda: tt_input(64, 64, device="TT").realize().min(axis=0).realize()),
  ("min_axis1",    lambda: tt_input(64, 64, device="TT").realize().min(axis=1).realize()),
  ("bcast_row",    lambda: ((lambda a, b: (a + b).realize())(tt_input(64, 64, device="TT").realize(), tt_input(64, 1, device="TT").realize()))),
  ("bcast_col",    lambda: ((lambda a, b: (a + b).realize())(tt_input(64, 64, device="TT").realize(), tt_input(1, 64, device="TT").realize()))),
  ("bcast_scalar", lambda: ((lambda a, b: (a * b).realize())(tt_input(64, 64, device="TT").realize(), tt_input(1, 1, device="TT").realize()))),
  ("fma",          lambda: ((lambda a, b, c: (a * b + c).realize())(
                      tt_input(64, 64, device="TT").realize(),
                      tt_input(64, 64, device="TT").realize(),
                      tt_input(64, 64, device="TT").realize()))),
  ("four_input",   lambda: ((lambda a, b, c, d: (a * b + c * d).realize())(
                      tt_input(64, 64, device="TT").realize(),
                      tt_input(64, 64, device="TT").realize(),
                      tt_input(64, 64, device="TT").realize(),
                      tt_input(64, 64, device="TT").realize()))),
  ("mlp_block",    lambda: ((lambda x, w1, w2: ((x @ w1).relu() @ w2).realize())(
                      tt_input(64, 64, device="TT").realize(),
                      tt_input(64, 64, device="TT").realize(),
                      tt_input(64, 64, device="TT").realize()))),
  ("relu_48",      lambda: tt_input(48, 48, device="TT").realize().relu().realize()),
  ("add_48",       lambda: ((lambda a, b: (a + b).realize())(tt_input(48, 48, device="TT").realize(), tt_input(48, 48, device="TT").realize()))),
  ("matmul_48",    lambda: ((lambda a, b: (a @ b).realize())(tt_input(48, 48, device="TT").realize(), tt_input(48, 48, device="TT").realize()))),
  ("sum_48_ax0",   lambda: tt_input(48, 48, device="TT").realize().sum(axis=0).realize()),
  ("relu_50_70",   lambda: tt_input(50, 70, device="TT").realize().relu().realize()),
  ("matmul_50_40_70", lambda: ((lambda a, b: (a @ b).realize())(tt_input(50, 40, device="TT").realize(), tt_input(40, 70, device="TT").realize()))),
  ("sum_70_ax1",   lambda: tt_input(50, 70, device="TT").realize().sum(axis=1).realize()),
  ("sum_full",     lambda: tt_input(64, 64, device="TT").realize().sum().realize()),
  ("max_full",     lambda: tt_input(64, 64, device="TT").realize().max().realize()),
  ("mean_full",    lambda: tt_input(64, 64, device="TT").realize().mean().realize()),
  ("sum_full_128", lambda: tt_input(128, 128, device="TT").realize().sum().realize()),
  ("sum_diff_full",lambda: ((lambda a, b: (a - b).sum().realize())(tt_input(64, 64, device="TT").realize(), tt_input(64, 64, device="TT").realize()))),
  ("sub_max",      lambda: ((lambda a: (a - a.max(axis=1, keepdim=True)).realize())(tt_input(64, 64, device="TT").realize()))),
  ("softmax_manual", lambda: ((lambda a: (
      lambda mx: (lambda e: (lambda s: (e / s).realize())(e.sum(axis=1, keepdim=True).realize()))((a - mx).exp().realize())
    )(a.max(axis=1, keepdim=True).realize()))(tt_input(64, 64, device="TT").realize()))),
  ("softmax_native", lambda: tt_input(64, 64, device="TT").realize().softmax(axis=1).realize()),
  ("layernorm_manual", lambda: ((lambda a: (
      lambda mean: (lambda c: (lambda sq: (lambda var: (lambda inv_std: (c * inv_std).realize())
        ((1.0 / (var + 1e-5).sqrt()).realize()))
        (sq.mean(axis=1, keepdim=True).realize()))
        ((c * c).realize()))
        ((a - mean).realize()))
        (a.mean(axis=1, keepdim=True).realize()))
      (tt_input(64, 64, device="TT").realize()))),
  ("layernorm_native", lambda: tt_input(64, 64, device="TT").realize().layernorm().realize()),
  ("layernorm_affine", lambda: ((lambda x, w, b: (x.layernorm() * w + b).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, device="TT").realize(),
      tt_input(64, device="TT").realize()))),
  ("softmax_ax0",  lambda: tt_input(64, 64, device="TT").realize().softmax(axis=0).realize()),
  ("var_ax1",      lambda: tt_input(64, 64, device="TT").realize().var(axis=1).realize()),
  ("std_ax1",      lambda: tt_input(64, 64, device="TT").realize().std(axis=1).realize()),
  ("matmul_bias",  lambda: ((lambda a, b, c: ((a @ b) + c).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, device="TT").realize()))),
  ("matmul_t",     lambda: ((lambda a, b: (a @ b.T).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("matmul_ta",    lambda: ((lambda a, b: (a.T @ b).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("matmul_t_relu",lambda: ((lambda a, b: ((a @ b.T).relu()).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("attn_qk",      lambda: ((lambda q, k: (((q @ k.T) * 0.125)).softmax(axis=1).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("attn_full",    lambda: ((lambda q, k, v: ((((q @ k.T) * 0.125).softmax(axis=1)).realize() @ v).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("mul_relu_chain", lambda: ((lambda a, b: (((a + b).realize() * 0.5).realize().relu()).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("matmul_t_128", lambda: ((lambda a, b: (a @ b.T).realize())(
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize()))),
  ("attn_full_128",lambda: ((lambda q, k, v: ((((q @ k.T) * 0.03125).softmax(axis=1)).realize() @ v).realize())(
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize()))),
  ("softmax_96",   lambda: tt_input(96, 96, device="TT").realize().softmax(axis=1).realize()),
  ("matmul_ta_rect", lambda: ((lambda a, b: (a.T @ b).realize())(
      tt_input(64, 128, device="TT").realize(),
      tt_input(64, 96, device="TT").realize()))),
  ("layernorm_96", lambda: tt_input(96, 96, device="TT").realize().layernorm().realize()),
  ("log_softmax",  lambda: tt_input(64, 64, device="TT").realize().log_softmax(axis=1).realize()),
  ("rms_norm",     lambda: ((lambda x: (x * (((x*x).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize())(
      tt_input(64, 64, device="TT").realize()))),
  ("softmax_after_matmul", lambda: ((lambda a, b: ((a @ b).realize() * 0.125).softmax(axis=1).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("swiglu",       lambda: ((lambda a, b: (a.silu() * b).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("silu_mlp",     lambda: ((lambda x, w1, w2: ((((x @ w1).realize().silu()).realize() @ w2).realize()))(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("silu_after_matmul", lambda: ((lambda a, b: ((a @ b).realize() * 0.015625).silu().realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("gelu_after_matmul_split", lambda: ((lambda a, b: (((a @ b).realize() * 0.015625).realize().gelu()).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("attn_masked",  lambda: ((lambda q, k, m: ((((q @ k.T) * 0.125) + m).softmax(axis=1)).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("layernorm_128", lambda: tt_input(128, 128, device="TT").realize().layernorm().realize()),
  ("rms_norm_128", lambda: ((lambda x: (x * (((x*x).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize())(
      tt_input(128, 128, device="TT").realize()))),
  ("swiglu_ffn",  lambda: ((lambda x, w1, w2, w3: (((x @ w1).realize().silu() * (x @ w2).realize()).realize() @ w3).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("mlp_residual", lambda: ((lambda x, w1, w2: (x + (((x @ w1).realize().relu()).realize() @ w2)).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("ln_residual",  lambda: ((lambda x, h: (x + h).realize().layernorm().realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("matmul_3way",  lambda: ((lambda a, b, c: (((a @ b).realize()) @ c).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("rmsnorm_mlp_block", lambda: ((lambda x, w1, w2: (x + (((((x * (((x*x).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize()) @ w1).realize().silu()).realize() @ w2)).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("matmul_rect_128", lambda: ((lambda a, b: (a @ b).realize())(
      tt_input(128, 64, device="TT").realize(),
      tt_input(64, 256, device="TT").realize()))),
  ("sigmoid_after_matmul", lambda: ((lambda a, b: ((a @ b).realize() * 0.015625).sigmoid().realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("transformer_attn_block", lambda: ((lambda q, k, v, x: (x + (((q @ k.T) * 0.125).softmax(axis=1)).realize() @ v).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("softmax_128",  lambda: tt_input(128, 128, device="TT").realize().softmax(axis=1).realize()),
  ("mean_full_128", lambda: tt_input(128, 128, device="TT").realize().mean().realize()),
  ("transformer_layer", lambda: ((lambda q, k, v, x, w1, w2:
      (lambda h: (h + ((((h * (((h*h).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize() @ w1).realize().silu()).realize() @ w2)).realize())(
        (x + (((q @ k.T) * 0.125).softmax(axis=1)).realize() @ v).realize()))(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("transformer_layer_128", lambda: ((lambda q, k, v, x, w1, w2:
      (lambda h: (h + ((((h * (((h*h).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize() @ w1).realize().silu()).realize() @ w2)).realize())(
        (x + (((q @ k.T) * 0.088388).softmax(axis=1)).realize() @ v).realize()))(
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize()))),
  ("rmsnorm_mlp_128", lambda: ((lambda x, w1, w2:
      (x + (((((x * (((x*x).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize()) @ w1).realize().silu()).realize() @ w2)).realize())(
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize()))),
  ("gelu_glu_split", lambda: ((lambda a, b: (a.gelu().realize() * b).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("gelu_glu_fused", lambda: ((lambda a, b: (a.gelu() * b).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("mlp_4x", lambda: ((lambda x, w1, w2: (((x @ w1).realize().silu()).realize() @ w2).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 256, device="TT").realize(),
      tt_input(256, 64, device="TT").realize()))),
  ("stacked_transformer", lambda: (lambda layer: (lambda q1, k1, v1, x, w1a, w1b, q2, k2, v2, w2a, w2b: layer(layer(x, q1, k1, v1, w1a, w1b), q2, k2, v2, w2a, w2b))(
      *[tt_input(64, 64, device="TT").realize() for _ in range(11)]))(
      lambda inp, q, k, v, wa, wb: (lambda h: (h + ((((h * (((h*h).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize() @ wa).realize().silu()).realize() @ wb)).realize())(
        (inp + (((q @ k.T) * 0.125).softmax(axis=1)).realize() @ v).realize()))),
  ("triple_transformer", lambda: (lambda layer: (lambda args: layer(layer(layer(args[3], args[0], args[1], args[2], args[4], args[5]), args[6], args[7], args[8], args[9], args[10]), args[11], args[12], args[13], args[14], args[15]))(
      [tt_input(64, 64, device="TT").realize() for _ in range(16)]))(
      lambda inp, q, k, v, wa, wb: (lambda h: (h + ((((h * (((h*h).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize() @ wa).realize().silu()).realize() @ wb)).realize())(
        (inp + (((q @ k.T) * 0.125).softmax(axis=1)).realize() @ v).realize()))),
  ("silu_mlp_residual", lambda: ((lambda x, w1, w2: (x + (((x @ w1).realize().silu()).realize() @ w2)).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("transformer_layer_4x", lambda: ((lambda q, k, v, x, w1, w2:
      (lambda h: (h + ((((h * (((h*h).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize() @ w1).realize().silu()).realize() @ w2)).realize())(
        (x + (((q @ k.T) * 0.125).softmax(axis=1)).realize() @ v).realize()))(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 256, device="TT").realize(),
      tt_input(256, 64, device="TT").realize()))),
  ("quad_transformer", lambda: (lambda layer: (lambda args: layer(layer(layer(layer(args[3], args[0], args[1], args[2], args[4], args[5]), args[6], args[7], args[8], args[9], args[10]), args[11], args[12], args[13], args[14], args[15]), args[16], args[17], args[18], args[19], args[20]))(
      [tt_input(64, 64, device="TT").realize() for _ in range(21)]))(
      lambda inp, q, k, v, wa, wb: (lambda h: (h + ((((h * (((h*h).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize() @ wa).realize().silu()).realize() @ wb)).realize())(
        (inp + (((q @ k.T) * 0.125).softmax(axis=1)).realize() @ v).realize()))),
  ("transformer_layer_4x_128", lambda: ((lambda q, k, v, x, w1, w2:
      (lambda h: (h + ((((h * (((h*h).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize() @ w1).realize().silu()).realize() @ w2)).realize())(
        (x + (((q @ k.T) * 0.088388).softmax(axis=1)).realize() @ v).realize()))(
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 128, device="TT").realize(),
      tt_input(128, 512, device="TT").realize(),
      tt_input(512, 128, device="TT").realize()))),
  ("cross_attention", lambda: ((lambda q, k, v: ((((q @ k.T) * 0.125).softmax(axis=1)).realize() @ v).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(96, 64, device="TT").realize(),
      tt_input(96, 64, device="TT").realize()))),
  ("l2_norm_rows", lambda: ((lambda x: ((x * x).sum(axis=1).sqrt()).realize())(
      tt_input(64, 64, device="TT").realize()))),
  ("mish", lambda: ((lambda x: (x * ((1 + x.exp()).log()).tanh()).realize())(
      tt_input(64, 64, device="TT").realize()))),
  ("hardswish", lambda: ((lambda x: (x * ((x + 3).maximum(0).minimum(6)) * (1/6)).realize())(
      tt_input(64, 64, device="TT").realize()))),
  ("positional_add", lambda: ((lambda x, p: (x + p).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("stacked_transformer_128", lambda: (lambda layer: (lambda q1, k1, v1, x, w1a, w1b, q2, k2, v2, w2a, w2b: layer(layer(x, q1, k1, v1, w1a, w1b), q2, k2, v2, w2a, w2b))(
      *[tt_input(128, 128, device="TT").realize() for _ in range(11)]))(
      lambda inp, q, k, v, wa, wb: (lambda h: (h + ((((h * (((h*h).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize() @ wa).realize().silu()).realize() @ wb)).realize())(
        (inp + (((q @ k.T) * 0.088388).softmax(axis=1)).realize() @ v).realize()))),
  ("cosine_sim_rows", lambda: ((lambda a, b: (
      (a * b).sum(axis=1).realize() / ((a * a).sum(axis=1).realize().sqrt().realize() * (b * b).sum(axis=1).realize().sqrt().realize())).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("geglu_ffn", lambda: ((lambda x, w1, w2, w3: ((((x @ w1).realize().gelu().realize() * (x @ w2).realize()).realize()) @ w3).realize())(
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize(),
      tt_input(64, 64, device="TT").realize()))),
  ("matmul_32", lambda: (tt_input(32, 32, device="TT") @ tt_input(32, 32, device="TT")).realize()),
  ("decoder_block", lambda: (lambda decoder: (lambda x, qs, ks, vs, qc, ek, ev, w1, w2: decoder(x, qs, ks, vs, qc, ek, ev, w1, w2))(
      *[tt_input(64, 64, device="TT").realize() for _ in range(9)]))(
      lambda x, qs, ks, vs, qc, ek, ev, w1, w2: (lambda h1: (lambda h2: (h2 + ((((h2 * (((h2*h2).mean(axis=1, keepdim=True) + 1e-5).rsqrt())).realize() @ w1).realize().silu()).realize() @ w2)).realize())(
        (h1 + (((qc @ ek.T) * 0.125).softmax(axis=1)).realize() @ ev).realize()))(
        (x + (((qs @ ks.T) * 0.125).softmax(axis=1)).realize() @ vs).realize()))),
]

REFS = {
  "add_mul":         (lambda a, b: (a + b) * 2,                                 [(DIM, DIM), (DIM, DIM)]),
  "relu":            (lambda a:    torch.relu(a),                               [(DIM, DIM)]),
  "mul":             (lambda a, b: a * b,                                       [(DIM, DIM), (DIM, DIM)]),
  "sqrt":            (lambda a:    torch.sqrt(a.clamp(min=0)),                  [(DIM, DIM)]),
  "sigmoid":         (lambda a:    torch.sigmoid(a),                            [(DIM, DIM)]),
  "matmul":          (lambda a, b: a @ b,                                       [(DIM, DIM), (DIM, DIM)]),
  "exp":             (lambda a:    torch.exp(a),                                [(DIM, DIM)]),
  "abs_diff":        (lambda a, b: (a - b).abs(),                               [(DIM, DIM), (DIM, DIM)]),
  "matmul_128":      (lambda a, b: a @ b,                                       [(128, 128), (128, 128)]),
  "matmul_rect":     (lambda a, b: a @ b,                                       [(64, 128), (128, 32)]),
  "neg":             (lambda a:    -a,                                          [(DIM, DIM)]),
  "recip":           (lambda a:    1.0 / (a + 1),                               [(DIM, DIM)]),
  "log":             (lambda a:    torch.log(a + 0.1),                          [(DIM, DIM)]),
  "tanh":            (lambda a:    torch.tanh(a),                               [(DIM, DIM)]),
  "matmul_256":      (lambda a, b: a @ b,                                       [(256, 256), (256, 256)]),
  "matmul_relu":     (lambda a, b: torch.relu(a @ b),                           [(DIM, DIM), (DIM, DIM)]),
  "matmul_add":      (lambda a, b, c: a @ b + c,                                [(DIM, DIM), (DIM, DIM), (DIM, DIM)]),
  "gelu":            (lambda a:    torch.nn.functional.gelu(a, approximate="tanh"), [(DIM, DIM)]),
  "silu":            (lambda a:    torch.nn.functional.silu(a),                 [(DIM, DIM)]),
  "square":          (lambda a:    a * a,                                       [(DIM, DIM)]),
  "rsqrt":           (lambda a:    torch.rsqrt(a + 0.1),                        [(DIM, DIM)]),
  "maximum":         (lambda a, b: torch.maximum(a, b),                         [(DIM, DIM), (DIM, DIM)]),
  "squared_diff":    (lambda a, b: (a - b) ** 2,                                [(DIM, DIM), (DIM, DIM)]),
  "matmul_silu":     (lambda a, b: torch.nn.functional.silu(a @ b),             [(DIM, DIM), (DIM, DIM)]),
  "div":             (lambda a, b: a / (b + 1),                                 [(DIM, DIM), (DIM, DIM)]),
  "sin":             (lambda a:    torch.sin(a),                                [(DIM, DIM)]),
  "cos":             (lambda a:    torch.cos(a),                                [(DIM, DIM)]),
  "minimum":         (lambda a, b: torch.minimum(a, b),                         [(DIM, DIM), (DIM, DIM)]),
  "scalar_add":      (lambda a:    a + 3.5,                                     [(DIM, DIM)]),
  "scalar_mul":      (lambda a:    a * 0.25,                                    [(DIM, DIM)]),
  "matmul_long_k":   (lambda a, b: a @ b,                                       [(64, 256), (256, 64)]),
  "matmul_thin_m":   (lambda a, b: a @ b,                                       [(32, 128), (128, 64)]),
  "sum_axis0":       (lambda a:    a.sum(dim=0),                                [(64, 64)], lambda r: r[0]),
  "max_axis0":       (lambda a:    a.max(dim=0).values,                         [(64, 64)], lambda r: r[0]),
  "sum_128":         (lambda a:    a.sum(dim=0),                                [(128, 128)], lambda r: r[0]),
  "sum_256_64":      (lambda a:    a.sum(dim=0),                                [(256, 64)], lambda r: r[0]),
  "sum_64_256":      (lambda a:    a.sum(dim=0),                                [(64, 256)], lambda r: r[0]),
  "sum_axis1":       (lambda a:    a.sum(dim=1),                                [(64, 64)], lambda r: r[:, 0]),
  "max_axis1":       (lambda a:    a.max(dim=1).values,                         [(64, 64)], lambda r: r[:, 0]),
  "sum_ax1_long":    (lambda a:    a.sum(dim=1),                                [(64, 256)], lambda r: r[:, 0]),
  "mean_axis0":      (lambda a:    a.mean(dim=0),                               [(64, 64)], lambda r: r[0]),
  "mean_axis1":      (lambda a:    a.mean(dim=1),                               [(64, 64)], lambda r: r[:, 0]),
  "min_axis0":       (lambda a:    a.min(dim=0).values,                         [(64, 64)], lambda r: r[0]),
  "min_axis1":       (lambda a:    a.min(dim=1).values,                         [(64, 64)], lambda r: r[:, 0]),
  "bcast_row":       (lambda a, b: a + b,                                       [(64, 64), (64, 1)]),
  "bcast_col":       (lambda a, b: a + b,                                       [(64, 64), (1, 64)]),
  "bcast_scalar":    (lambda a, b: a * b,                                       [(64, 64), (1, 1)]),
  "fma":             (lambda a, b, c: a * b + c,                                [(64, 64), (64, 64), (64, 64)]),
  "four_input":      (lambda a, b, c, d: a * b + c * d,                         [(64, 64), (64, 64), (64, 64), (64, 64)]),
  "relu_48":         (lambda a:    torch.relu(a),                               [(48, 48)]),
  "add_48":          (lambda a, b: a + b,                                       [(48, 48), (48, 48)]),
  "matmul_48":       (lambda a, b: a @ b,                                       [(48, 48), (48, 48)]),
  "sum_48_ax0":      (lambda a:    a.sum(dim=0),                                [(48, 48)], lambda r: r[0]),
  "relu_50_70":      (lambda a:    torch.relu(a),                               [(50, 70)]),
  "matmul_50_40_70": (lambda a, b: a @ b,                                       [(50, 40), (40, 70)]),
  "sum_70_ax1":      (lambda a:    a.sum(dim=1),                                [(50, 70)], lambda r: r[:, 0]),
  "sum_full":        (lambda a:    a.sum(),                                     [(64, 64)], lambda r: r[0, 0]),
  "max_full":        (lambda a:    a.max(),                                     [(64, 64)], lambda r: r[0, 0]),
  "mean_full":       (lambda a:    a.mean(),                                    [(64, 64)], lambda r: r[0, 0]),
  "sum_full_128":    (lambda a:    a.sum(),                                     [(128, 128)], lambda r: r[0, 0]),
  "sum_diff_full":   (lambda a, b: (a - b).sum(),                               [(64, 64), (64, 64)], lambda r: r[0, 0]),
  "sub_max":         (lambda a:    a - a.max(dim=1, keepdim=True).values,       [(64, 64)]),
  "softmax_manual":  (lambda a:    torch.softmax(a, dim=1),                     [(64, 64)]),
  "softmax_native":  (lambda a:    torch.softmax(a, dim=1),                     [(64, 64)]),
  "layernorm_manual":(lambda a:    (lambda c: c / torch.sqrt((c * c).mean(dim=1, keepdim=True) + 1e-5))(a - a.mean(dim=1, keepdim=True)),
                                                                                [(64, 64)]),
  "layernorm_native":(lambda a:    torch.nn.functional.layer_norm(a, (a.shape[-1],), eps=1e-5),
                                                                                [(64, 64)]),
  "layernorm_affine":(lambda x, w, b: torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=w, bias=b, eps=1e-5),
                                                                                [(64, 64), ROW64, ROW64]),
  "softmax_ax0":     (lambda a: torch.softmax(a, dim=0),                        [(64, 64)]),
  "var_ax1":         (lambda a: a.var(dim=1, keepdim=True, unbiased=False),     [(64, 64)]),
  "std_ax1":         (lambda a: a.std(dim=1, keepdim=True, unbiased=False),     [(64, 64)]),
  "matmul_bias":     (lambda a, b, c: a @ b + c,                                [(64, 64), (64, 64), ROW64]),
  "matmul_t":        (lambda a, b: a @ b.T,                                     [(64, 64), (64, 64)]),
  "matmul_ta":       (lambda a, b: a.T @ b,                                     [(64, 64), (64, 64)]),
  "matmul_t_relu":   (lambda a, b: torch.relu(a @ b.T),                         [(64, 64), (64, 64)]),
  "attn_qk":         (lambda q, k: torch.softmax((q @ k.T) * 0.125, dim=1),     [(64, 64), (64, 64)]),
  "attn_full":       (lambda q, k, v: torch.softmax((q @ k.T) * 0.125, dim=1) @ v, [(64, 64), (64, 64), (64, 64)]),
  "mul_relu_chain":  (lambda a, b: torch.relu((a + b) * 0.5),                   [(64, 64), (64, 64)]),
  "matmul_t_128":    (lambda a, b: a @ b.T,                                     [(128, 128), (128, 128)]),
  "attn_full_128":   (lambda q, k, v: torch.softmax((q @ k.T) * 0.03125, dim=1) @ v, [(128, 128), (128, 128), (128, 128)]),
  "softmax_96":      (lambda a: torch.softmax(a, dim=1),                        [(96, 96)]),
  "matmul_ta_rect":  (lambda a, b: a.T @ b,                                     [(64, 128), (64, 96)]),
  "layernorm_96":    (lambda a: torch.nn.functional.layer_norm(a, (a.shape[-1],), eps=1e-5), [(96, 96)]),
  "log_softmax":     (lambda a: torch.log_softmax(a, dim=1),                    [(64, 64)]),
  "rms_norm":        (lambda x: x * torch.rsqrt((x*x).mean(dim=1, keepdim=True) + 1e-5), [(64, 64)]),
  "softmax_after_matmul": (lambda a, b: torch.softmax((a @ b) * 0.125, dim=1), [(64, 64), (64, 64)]),
  "swiglu":               (lambda a, b: torch.nn.functional.silu(a) * b,         [(64, 64), (64, 64)]),
  "silu_mlp":             (lambda x, w1, w2: torch.nn.functional.silu(x @ w1) @ w2, [(64, 64), (64, 64), (64, 64)]),
  "silu_after_matmul":    (lambda a, b: torch.nn.functional.silu((a @ b) * 0.015625), [(64, 64), (64, 64)]),
  "gelu_after_matmul_split": (lambda a, b: torch.nn.functional.gelu((a @ b) * 0.015625, approximate="tanh"), [(64, 64), (64, 64)]),
  "attn_masked":          (lambda q, k, m: torch.softmax((q @ k.T) * 0.125 + m, dim=1), [(64, 64), (64, 64), (64, 64)]),
  "layernorm_128":        (lambda a: torch.nn.functional.layer_norm(a, (a.shape[-1],), eps=1e-5), [(128, 128)]),
  "rms_norm_128":         (lambda x: x * torch.rsqrt((x*x).mean(dim=1, keepdim=True) + 1e-5), [(128, 128)]),
  "swiglu_ffn":           (lambda x, w1, w2, w3: (torch.nn.functional.silu(x @ w1) * (x @ w2)) @ w3, [(64, 64), (64, 64), (64, 64), (64, 64)]),
  "mlp_residual":         (lambda x, w1, w2: x + torch.relu(x @ w1) @ w2, [(64, 64), (64, 64), (64, 64)]),
  "ln_residual":          (lambda x, h: torch.nn.functional.layer_norm(x + h, ((x+h).shape[-1],), eps=1e-5), [(64, 64), (64, 64)]),
  "matmul_3way":          (lambda a, b, c: (a @ b) @ c, [(64, 64), (64, 64), (64, 64)]),
  "rmsnorm_mlp_block":    (lambda x, w1, w2: x + torch.nn.functional.silu((x * torch.rsqrt((x*x).mean(dim=1, keepdim=True) + 1e-5)) @ w1) @ w2, [(64, 64), (64, 64), (64, 64)]),
  "matmul_rect_128":      (lambda a, b: a @ b, [(128, 64), (64, 256)]),
  "sigmoid_after_matmul": (lambda a, b: torch.sigmoid((a @ b) * 0.015625), [(64, 64), (64, 64)]),
  "transformer_attn_block": (lambda q, k, v, x: x + torch.softmax((q @ k.T) * 0.125, dim=1) @ v, [(64, 64), (64, 64), (64, 64), (64, 64)]),
  "softmax_128":          (lambda a: torch.softmax(a, dim=1), [(128, 128)]),
  "mean_full_128":        (lambda a: a.mean(), [(128, 128)], lambda r: r[0, 0]),
  "transformer_layer":    (lambda q, k, v, x, w1, w2: (lambda h: h + torch.nn.functional.silu((h * torch.rsqrt((h*h).mean(dim=1, keepdim=True) + 1e-5)) @ w1) @ w2)(x + torch.softmax((q @ k.T) * 0.125, dim=1) @ v), [(64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64)]),
  "transformer_layer_128":(lambda q, k, v, x, w1, w2: (lambda h: h + torch.nn.functional.silu((h * torch.rsqrt((h*h).mean(dim=1, keepdim=True) + 1e-5)) @ w1) @ w2)(x + torch.softmax((q @ k.T) * 0.088388, dim=1) @ v), [(128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128)]),
  "rmsnorm_mlp_128":      (lambda x, w1, w2: x + torch.nn.functional.silu((x * torch.rsqrt((x*x).mean(dim=1, keepdim=True) + 1e-5)) @ w1) @ w2, [(128, 128), (128, 128), (128, 128)]),
  "gelu_glu_split":       (lambda a, b: torch.nn.functional.gelu(a, approximate="tanh") * b, [(64, 64), (64, 64)]),
  "gelu_glu_fused":       (lambda a, b: torch.nn.functional.gelu(a, approximate="tanh") * b, [(64, 64), (64, 64)]),
  "mlp_4x":               (lambda x, w1, w2: torch.nn.functional.silu(x @ w1) @ w2, [(64, 64), (64, 256), (256, 64)]),
  "stacked_transformer":  (lambda q1, k1, v1, x, w1a, w1b, q2, k2, v2, w2a, w2b: (lambda h: h + torch.nn.functional.silu((h * torch.rsqrt((h*h).mean(dim=1, keepdim=True) + 1e-5)) @ w2a) @ w2b)((lambda layer1: layer1 + torch.softmax((q2 @ k2.T) * 0.125, dim=1) @ v2)((lambda h: h + torch.nn.functional.silu((h * torch.rsqrt((h*h).mean(dim=1, keepdim=True) + 1e-5)) @ w1a) @ w1b)(x + torch.softmax((q1 @ k1.T) * 0.125, dim=1) @ v1))), [(64, 64)] * 11),
  "triple_transformer":   ((lambda layer: lambda q1, k1, v1, x, w1a, w1b, q2, k2, v2, w2a, w2b, q3, k3, v3, w3a, w3b: layer(layer(layer(x, q1, k1, v1, w1a, w1b), q2, k2, v2, w2a, w2b), q3, k3, v3, w3a, w3b))(lambda inp, q, k, v, wa, wb: (lambda h: h + torch.nn.functional.silu((h * torch.rsqrt((h*h).mean(dim=1, keepdim=True) + 1e-5)) @ wa) @ wb)(inp + torch.softmax((q @ k.T) * 0.125, dim=1) @ v)), [(64, 64)] * 16),
  "silu_mlp_residual":    (lambda x, w1, w2: x + torch.nn.functional.silu(x @ w1) @ w2, [(64, 64), (64, 64), (64, 64)]),
  "transformer_layer_4x": (lambda q, k, v, x, w1, w2: (lambda h: h + torch.nn.functional.silu((h * torch.rsqrt((h*h).mean(dim=1, keepdim=True) + 1e-5)) @ w1) @ w2)(x + torch.softmax((q @ k.T) * 0.125, dim=1) @ v), [(64, 64), (64, 64), (64, 64), (64, 64), (64, 256), (256, 64)]),
  "quad_transformer":     ((lambda layer: lambda q1, k1, v1, x, w1a, w1b, q2, k2, v2, w2a, w2b, q3, k3, v3, w3a, w3b, q4, k4, v4, w4a, w4b: layer(layer(layer(layer(x, q1, k1, v1, w1a, w1b), q2, k2, v2, w2a, w2b), q3, k3, v3, w3a, w3b), q4, k4, v4, w4a, w4b))(lambda inp, q, k, v, wa, wb: (lambda h: h + torch.nn.functional.silu((h * torch.rsqrt((h*h).mean(dim=1, keepdim=True) + 1e-5)) @ wa) @ wb)(inp + torch.softmax((q @ k.T) * 0.125, dim=1) @ v)), [(64, 64)] * 21),
  "transformer_layer_4x_128": (lambda q, k, v, x, w1, w2: (lambda h: h + torch.nn.functional.silu((h * torch.rsqrt((h*h).mean(dim=1, keepdim=True) + 1e-5)) @ w1) @ w2)(x + torch.softmax((q @ k.T) * 0.088388, dim=1) @ v), [(128, 128), (128, 128), (128, 128), (128, 128), (128, 512), (512, 128)]),
  "cross_attention":      (lambda q, k, v: torch.softmax((q @ k.T) * 0.125, dim=1) @ v, [(64, 64), (96, 64), (96, 64)]),
  "l2_norm_rows":         (lambda x: torch.sqrt((x * x).sum(dim=1)), [(64, 64)], lambda r: r[:, 0]),
  "mish":                 (lambda x: x * torch.tanh(torch.log(1 + torch.exp(x))), [(64, 64)]),
  "hardswish":            (lambda x: x * torch.clamp(x + 3, min=0, max=6) * (1/6), [(64, 64)]),
  "positional_add":       (lambda x, p: x + p, [(64, 64), (64, 64)]),
  "stacked_transformer_128": ((lambda layer: lambda q1, k1, v1, x, w1a, w1b, q2, k2, v2, w2a, w2b: layer(layer(x, q1, k1, v1, w1a, w1b), q2, k2, v2, w2a, w2b))(lambda inp, q, k, v, wa, wb: (lambda h: h + torch.nn.functional.silu((h * torch.rsqrt((h*h).mean(dim=1, keepdim=True) + 1e-5)) @ wa) @ wb)(inp + torch.softmax((q @ k.T) * 0.088388, dim=1) @ v)), [(128, 128)] * 11),
  "cosine_sim_rows":      (lambda a, b: (a * b).sum(dim=1) / (torch.sqrt((a * a).sum(dim=1)) * torch.sqrt((b * b).sum(dim=1))), [(64, 64), (64, 64)], lambda r: r[:, 0]),
  "geglu_ffn":            (lambda x, w1, w2, w3: (torch.nn.functional.gelu(x @ w1, approximate="tanh") * (x @ w2)) @ w3, [(64, 64)] * 4),
  "matmul_32":            (lambda a, b: a @ b, [(32, 32), (32, 32)]),
  "decoder_block":        (lambda qs, ks, vs, x, qc, ek, ev, w1, w2: (lambda h2: h2 + torch.nn.functional.silu((h2 * torch.rsqrt((h2*h2).mean(dim=1, keepdim=True) + 1e-5)) @ w1) @ w2)((lambda h1: h1 + torch.softmax((qc @ ek.T) * 0.125, dim=1) @ ev)(x + torch.softmax((qs @ ks.T) * 0.125, dim=1) @ vs)), [(64, 64)] * 9),
  "mlp_block":       (lambda x, w1, w2: torch.relu(x @ w1) @ w2,                [(64, 64), (64, 64), (64, 64)]),
}
