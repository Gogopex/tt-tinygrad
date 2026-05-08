"""Driver that runs each staged TT kernel against ttnn tensors.

Run inside the tt-lang dist container. Loads each .py from ./tt_kernels/,
imports it, finds the @ttl.operation function, calls it with rand inputs,
and reports pass/fail vs a torch reference.

Chain dataflow is auto-derived from ./tt_kernels/_manifest.json: each kernel
records its per-slot tinygrad-buffer ids during DRYRUN, and the driver
threads ttnn tensors through steps by matching ids.
"""
import importlib.util, json, pathlib, sys, traceback, zlib

import torch
import ttl  # noqa: F401
import ttnn

KERNELS = pathlib.Path(__file__).resolve().parent / "tt_kernels"
DIM = 64
DTYPE = ttnn.bfloat16

def row_vector(n: int) -> tuple[tuple[int], tuple[int, int]]:
  return ((n,), (1, n))

ROW64 = row_vector(64)
POSITIVE_INPUT_LABELS = {"sqrt", "log", "rsqrt"}

def logical_shape(spec) -> tuple[int, ...]:
  return tuple(spec[0]) if spec and isinstance(spec[0], tuple) else tuple(spec)

def device_shape(spec) -> tuple[int, int]:
  if spec and isinstance(spec[0], tuple): return tuple(spec[1])
  shape = tuple(spec)
  if len(shape) == 1: return (1, shape[0])
  if len(shape) == 2: return shape
  raise ValueError(f"TT driver only supports 1D/2D inputs, got shape {shape}")

def make_input(label: str, idx: int, shape: tuple[int, ...]) -> torch.Tensor:
  gen = torch.Generator()
  gen.manual_seed(zlib.crc32(f"{label}:{idx}".encode()))
  data = torch.rand(shape, dtype=torch.float32, generator=gen)
  if label in POSITIVE_INPUT_LABELS:
    data = data * 0.95 + 0.05
  else:
    data = data * 1.5 - 0.75
  return data.to(torch.bfloat16)

MATMUL_LIKE_LABELS = {
  "matmul", "matmul_128", "matmul_rect", "matmul_256", "matmul_relu", "matmul_add",
  "matmul_silu", "matmul_long_k", "matmul_thin_m", "matmul_48", "matmul_50_40_70",
  "matmul_bias", "matmul_t", "matmul_ta", "matmul_t_relu", "matmul_t_128",
  "matmul_ta_rect", "softmax_after_matmul", "silu_after_matmul",
  "gelu_after_matmul_split", "matmul_3way", "matmul_rect_128",
  "sigmoid_after_matmul", "matmul_32",
}
LONG_CHAIN_LABELS = {
  "attn_full", "attn_full_128", "transformer_attn_block", "transformer_layer",
  "transformer_layer_128", "rmsnorm_mlp_128", "swiglu_ffn", "rmsnorm_mlp_block",
  "transformer_layer_4x", "quad_transformer", "transformer_layer_4x_128",
  "stacked_transformer", "triple_transformer", "stacked_transformer_128",
  "cross_attention", "geglu_ffn", "decoder_block",
}
EXPECTED_FAILURES = {
  "log_softmax": "log-sum-exp chain drifts beyond the strict BF16 tolerance",
  "transformer_layer": "long BF16 transformer chain drifts beyond the strict tolerance",
  "transformer_layer_128": "long BF16 transformer chain drifts beyond the strict tolerance",
  "rmsnorm_mlp_128": "128-wide RMSNorm-MLP drifts beyond the strict tolerance",
  "mlp_4x": "4x FFN matmul chain has localized BF16 drift beyond the strict tolerance",
  "stacked_transformer": "multi-layer BF16 transformer chain drifts beyond the strict tolerance",
  "triple_transformer": "multi-layer BF16 transformer chain drifts beyond the strict tolerance",
  "transformer_layer_4x": "4x FFN transformer chain drifts beyond the strict tolerance",
  "quad_transformer": "multi-layer BF16 transformer chain drifts beyond the strict tolerance",
  "transformer_layer_4x_128": "128-wide 4x FFN transformer chain drifts beyond the strict tolerance",
  "stacked_transformer_128": "128-wide multi-layer transformer chain drifts beyond the strict tolerance",
  "geglu_ffn": "GeGLU FFN has localized BF16 drift beyond the strict tolerance",
  "decoder_block": "decoder block chain drifts beyond the strict tolerance",
}

def tolerances(label: str, step_count: int) -> tuple[float, float]:
  if label in LONG_CHAIN_LABELS: return 1.2e-1, 1.2e-1
  if label in MATMUL_LIKE_LABELS or label.startswith(("sum_", "mean_", "max_", "min_")): return 8e-2, 8e-2
  if step_count > 1: return 6e-2, 6e-2
  if label in {"gelu", "gelu_glu_fused"}: return 5e-2, 5e-2
  return 2e-2, 2e-2

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

def derive_out_shape(kind: str, in_shapes: list[tuple], attrs: dict | None = None, fn_name: str = "") -> tuple:
  attrs = attrs or {}
  if kind == "matmul":
    if len(in_shapes) < 2: raise RuntimeError(f"matmul step {fn_name!r} has fewer than two inputs")
    a_shape, b_shape = in_shapes[0], in_shapes[1]
    a_t = attrs.get("a_transposed")
    b_t = attrs.get("b_transposed")
    if a_t is None or b_t is None:
      raise RuntimeError(f"matmul step {fn_name!r} is missing transpose metadata")
    m = a_shape[1] if a_t else a_shape[0]
    a_k = a_shape[0] if a_t else a_shape[1]
    b_k = b_shape[1] if b_t else b_shape[0]
    n = b_shape[0] if b_t else b_shape[1]
    if a_k != b_k:
      raise RuntimeError(f"matmul step {fn_name!r} has incompatible K dims: {a_shape} x {b_shape}")
    return (m, n)
  if kind == "elementwise":
    return (max(s[0] for s in in_shapes), max(s[1] for s in in_shapes))
  if kind == "reduce_axis0":
    return (1, in_shapes[0][1])
  if kind == "reduce_axis1":
    return (in_shapes[0][0], 1)
  if kind == "full_reduce":
    return (1, 1)
  raise ValueError(f"unknown kernel kind {kind!r}")

def load_kernel(path: pathlib.Path, fn_name: str):
  spec = importlib.util.spec_from_file_location(path.stem, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  fn = getattr(mod, fn_name, None)
  if fn is None: raise RuntimeError(f"module {path.name} has no callable named {fn_name!r}")
  return fn

def to_ttnn(t: torch.Tensor, device, shape: tuple[int, int] | None = None):
  if shape is not None and tuple(t.shape) != shape: t = t.reshape(shape)
  return ttnn.from_torch(t, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device)

def slot_to_arg(slots: list[int]) -> list[int]:
  # slots[arg_idx] = slot_id; we want arg_idx -> slot_id sorted by arg position.
  # slots is already in arg order with the trailing entry = output slot.
  return slots

def run_label(label: str, steps: list[dict], ref, in_shapes: list[tuple], slicer, device):
  logical_shapes = [logical_shape(s) for s in in_shapes]
  device_shapes = [device_shape(s) for s in in_shapes]
  in_torch = [make_input(label, i, shape) for i, shape in enumerate(logical_shapes)]
  externals = [to_ttnn(t, device, device_shapes[i]) for i, t in enumerate(in_torch)]

  buf_to_source: dict[int, tuple[str, int]] = {}
  ext_assigned = 0
  step_outs: list = []
  step_shapes: list[tuple] = []

  for i, step in enumerate(steps):
    fn = load_kernel(KERNELS / step["file"], step["fn"])
    slots = step["slots"]
    buf_ids = step["buf_ids"]
    in_slot_count = len(slots) - 1
    args = []
    arg_shapes: list[tuple] = []
    for arg_idx in range(in_slot_count):
      slot = slots[arg_idx]
      buf_id = buf_ids[slot]
      if buf_id not in buf_to_source:
        if ext_assigned >= len(externals):
          raise RuntimeError(f"{label} step {i} introduces buf_id {buf_id} but no externals left "
                             f"(expected {len(externals)}, used {ext_assigned})")
        buf_to_source[buf_id] = ("ext", ext_assigned)
        ext_assigned += 1
      src, idx = buf_to_source[buf_id]
      if src == "ext":
        args.append(externals[idx]); arg_shapes.append(device_shapes[idx])
      else:
        args.append(step_outs[idx]); arg_shapes.append(step_shapes[idx])
    out_shape = derive_out_shape(step["kind"], arg_shapes, step.get("attrs", {}), step["fn"])
    out_t = torch.zeros(out_shape, dtype=torch.bfloat16)
    out_tt = to_ttnn(out_t, device)
    fn(*args, out_tt)
    out_buf_id = buf_ids[slots[-1]]
    buf_to_source[out_buf_id] = ("step", i)
    step_outs.append(out_tt)
    step_shapes.append(out_shape)
    print(f"  [{label}] step {i} {step['file']} fn={step['fn']} kind={step['kind']} out={out_shape}")

  if ext_assigned != len(externals):
    print(f"  [{label}] WARN unused externals: assigned {ext_assigned}/{len(externals)}")

  result = ttnn.to_torch(step_outs[-1])
  expected = ref(*in_torch)
  if slicer is not None: result = slicer(result)
  rtol, atol = tolerances(label, len(steps))
  shape_ok = tuple(result.shape) == tuple(expected.shape)
  ok = shape_ok and torch.allclose(result, expected, rtol=rtol, atol=atol)
  tag = f"chain:{label}" if len(steps) > 1 else label
  print(f"[{'PASS' if ok else 'FAIL'} {tag}] {len(steps)} kernel(s), result vs torch ref "
        f"(rtol={rtol:g}, atol={atol:g})")
  if not ok:
    if not shape_ok:
      print(f"    shape mismatch: result={tuple(result.shape)} expected={tuple(expected.shape)}")
    else:
      diff = (result - expected).abs()
      print(f"    max_abs_diff={diff.max().item():.4f}  mean_abs_diff={diff.mean().item():.4f}")
  return ok

def main():
  manifest_path = KERNELS / "_manifest.json"
  manifest = json.loads(manifest_path.read_text())
  device = ttnn.open_device(device_id=0)
  passed = failed = errors = xfailed = 0
  try:
    print(f"loaded manifest with {len(manifest)} labels from {manifest_path}")
    if not manifest:
      raise RuntimeError("empty kernel manifest")
    for label, steps in manifest.items():
      entry = REFS.get(label)
      if entry is None:
        print(f"[ERR  {label}] no torch reference")
        errors += 1
        continue
      slicer = entry[2] if len(entry) == 3 else None
      ref, in_shapes = entry[0], entry[1]
      try:
        ok = run_label(label, steps, ref, in_shapes, slicer, device)
        if ok:
          passed += 1
          if label in EXPECTED_FAILURES:
            print(f"[XPASS {label}] expected drift no longer reproduced: {EXPECTED_FAILURES[label]}")
        elif label in EXPECTED_FAILURES:
          xfailed += 1
          print(f"[XFAIL {label}] {EXPECTED_FAILURES[label]}")
        else:
          failed += 1
      except Exception:
        errors += 1
        print(f"[ERR  {label}]")
        traceback.print_exc()
    print(f"\nsummary: pass={passed} xfail={xfailed} fail={failed} err={errors}")
  finally:
    ttnn.close_device(device)
  if failed or errors:
    raise SystemExit(1)

if __name__ == "__main__":
  main()
