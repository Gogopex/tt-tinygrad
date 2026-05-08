"""Generate tinygrad TT kernels and stage them for execution on a TT host.

Renders each test case as its own importable Python file under ./tt_kernels/
and writes a single driver script that imports and invokes them with ttnn
tensors. The driver is expected to run inside the tt-lang dist container
(or any host with ttl + ttnn installed).
"""
import os, sys, io, contextlib, pathlib, shutil, re, json

REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "tinygrad"))
os.environ.setdefault("TT_DRYRUN", "1")
os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")

from tinygrad import Tensor
import tinygrad.runtime.ops_tt as ops_tt

def tt_input(*shape, device="TT", dtype=None):
  return Tensor.empty(*shape, device=device, dtype=dtype)

ALLOW_KERNEL_SKIPS = os.environ.get("TT_ALLOW_KERNEL_SKIPS") == "1"

OUT = REPO / "tt_kernels"
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir()

def capture(thunk):
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf): thunk()
  return buf.getvalue()

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

def _skip(name: str, src: str) -> bool:
  if "NotImplementedError" in src: return True
  if re.match(r"^E_2(n\d*)?$", name): return True
  return False

manifest = []
chain_manifest: dict[str, list[dict]] = {}
for label, thunk in CASES:
  ops_tt._dryrun_calls.clear()
  capture(thunk)
  calls = list(ops_tt._dryrun_calls)
  step = 0
  for call in calls:
    name, src, buf_ids = call["name"], call["src"], call["buf_ids"]
    if _skip(name, src):
      msg = f"{label}: renderer produced unsupported/skipped kernel {name!r}"
      if not ALLOW_KERNEL_SKIPS: raise RuntimeError(msg)
      print(f"  skipped {name}: {msg}")
      continue
    contract = call.get("contract")
    if contract is None:
      raise RuntimeError(f"kernel {name} for {label} missing tt_contract metadata")
    if contract.get("version") != 1:
      raise RuntimeError(f"kernel {name} for {label} has unsupported contract version {contract.get('version')!r}")
    fname = f"{label}__{step:02d}_{name}.py"
    (OUT / fname).write_text(src)
    manifest.append((label, name, fname))
    chain_manifest.setdefault(label, []).append({
      "file": fname, "fn": name, "buf_ids": buf_ids, **contract,
    })
    print(f"  wrote {fname}")
    step += 1
  if label not in chain_manifest:
    raise RuntimeError(f"{label}: no kernels staged")

(OUT / "_manifest.json").write_text(json.dumps(chain_manifest, indent=2))
print(f"\n{len(manifest)} kernels staged in {OUT}")
print(f"manifest: {OUT / '_manifest.json'}")
for label, name, fname in manifest:
  print(f"  {label:10s} -> {fname}")
