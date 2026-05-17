# tinygrad-tt

A Tenstorrent backend for [tinygrad](https://github.com/tinygrad/tinygrad).
Targets [TT-Lang](https://github.com/tenstorrent/tt-lang) (`ttl.math.*`) as the codegen layer instead of raw tt-metal, and runs the resulting kernels through [ttnn](https://github.com/tenstorrent/tt-metal) on a Wormhole / QuietBox host.

Status: PoC. On QuietBox with `tt-lang-dist` 1.0.6 `python tt_tests.py` reports `pass=110 xfail=13 skip=2 fail=0 err=0` over 125 cases (the 2 skips are `matmul_bias` / `layernorm_affine`, which need 1D inputs that the V0 runtime doesn't yet plumb).

## The stack, in one paragraph

`tinygrad` lowers `Tensor.foo()` into a graph of UOps and then asks a per-device `Renderer` to turn each kernel's UOps into source code, which a per-device `Compiled` backend executes. `TT-Lang` is a Python DSL for tile-program kernels: you write functions that consume / produce `ttnn` tile tensors using `ttl.math.*` ops, and `ttl` arranges for them to run on a Tenstorrent card via the `ttnn` runtime. **This repo is the glue**: a `Renderer` that emits TT-Lang Python source for each UOp graph, and a `Compiled` device that parses the rendered kernel's contract, materializes `ttnn` tensors from host buffers, and calls the kernel in-process.

## What this repo adds

A three-line patch to upstream tinygrad, plus a small set of Python files:

| File | Role |
|---|---|
| `patches/0001-tt-device-hooks.patch` | 3 trivial hooks in tinygrad: register `"TT"` in `ALL_DEVICES`, add a `disable_hand_coded_opts` flag on `Renderer`, honor it in `postrange.py` (the tile-natural shapes we want would be wrecked by tinygrad's heuristic opt pass) |
| `tt_renderer.py` | The `Renderer`. Classifies each kernel as matmul / reduce / full-reduce / elementwise from its UOps, then emits TT-Lang Python source. Embeds a `# tt_contract: {...}` JSON header in every kernel describing kind, input/output slot indices, and attrs (matmul transpose flags, reduce axis). Symlinked into the tinygrad tree as `tinygrad/renderer/tt.py`. |
| `tt_device.py` | The `Compiled` device. Parses each rendered kernel's contract, lazily materializes `ttnn` tensors from host buffers, calls the kernel, and marks the output for lazy copy-out. Symlinked as `tinygrad/runtime/ops_tt.py`. |
| `tt_runtime.py` | User-facing helpers: `from_torch(t)` → tinygrad Tensor on `"TT"` with shape/dtype registered on the underlying buffer; `to_torch(t)` ← realized Tensor, syncing ttnn → host on demand. |
| `tt_test_cases.py` | The 125 test cases as (label, thunk) pairs plus their torch references. |
| `tt_tests.py` | End-to-end test driver: builds deterministic bf16 inputs, runs each case via `.realize()`, compares against torch. |
| `tt_runtime_smoke.py` | 36-test smoke for the in-process runtime. |

`scripts/setup.sh` clones tinygrad at `TINYGRAD_PIN`, applies the patch, and symlinks the renderer and device into the tinygrad tree.

## Execution flow

```
host (TT-Lang SDK installed):

  tt_runtime.from_torch(t)             # host bytes registered as TT buffer
  → normal tinygrad ops: a + b, a @ b, .softmax(), ...
  → .realize() triggers TTRenderer → TTProgram.__call__
       → parse contract, lazy-materialize ttnn inputs
       → call the rendered kernel
       → mark output buffer dirty (lazy copy-out)
  tt_runtime.to_torch(t)               # ttnn → host on demand
```

## What works

- **Elementwise**: relu, sigmoid, gelu, silu, tanh, sin, cos, exp, log, sqrt, rsqrt, square, neg, recip, abs (via sign), scalar add/mul, div, maximum, minimum, squared_diff, fma (`a*b+c`), four-input (`a*b+c*d`).
- **Matmul**: square (64/128/256), rectangular, long-K, thin-M, transposed (`a@b.T`, `a.T@b`), with `+relu` / `+silu` / `+add` / `+bias` / `+scale` fusion.
- **Attention**: `softmax(Q@K.T / √d, axis=1)` and full `softmax(Q@K.T / √d) @ V`, at 64×64 and 128×128, plus residual attention `x + softmax(...) @ V`.
- **Generic chains**: e.g. `relu((a + b) * 0.5)` lowers as 3 separate elementwise kernels with auto-derived dataflow.
- **Reductions**: sum / max / min / mean on axis 0 or 1, plus full-tensor reductions to scalar; `(a-b).sum()` fused.
- **Broadcast**: row, col, scalar; axis-aware extra-input broadcast via partial `ttl.math.broadcast`.
- **Non-tile-aligned shapes**: e.g. (48,48), (50,70), (50,40)@(40,70), softmax/layernorm at (96,96) via TILE_LAYOUT padding + ceil-div tile loops.
- **Native fused-reduce-with-prologue-and-epilogue**: `Tensor.softmax(axis=1)`, `Tensor.layernorm()`, and `rms_norm = x * rsqrt(mean(x²)+ε)` lower into multi-kernel chains and validate.
- **Transformer-adjacent chains**: scaled-dot-product attention, attention with residual, cross-attention with rectangular Q/KV, SwiGLU, SiLU MLP, masked attention logits, residual MLP blocks, `(a@b)@c`, L2 norm, mish, hardswish, positional add, cosine similarity.

## What doesn't work yet

- **argmax / argmin**: blocked at the `ttl` primitive layer. Hardware-mode `ttl.math` exposes neither an iota/arange tile generator nor elementwise `eq`/`where` (only the simulator has `where`). Not fixable in the renderer without upstream `ttl` primitives.
- **`Tensor.rand(device="TT")`**: tinygrad's threefry RNG uses bitwise ops the tile renderer doesn't handle. The staged path works around this with explicit `tt_input(...)` placeholders; the host driver fills them from deterministic torch seeds.
- **bf16-on-tile, fp32-in-buffer, 2D-only**: the in-process runtime doesn't plumb higher-rank tensors or dtype control yet.
- **13 known numerical-drift cases**: marked XFAIL by the driver under strict bf16 tolerance — `log_softmax`, `mlp_4x`, `geglu_ffn`, `decoder_block`, and the longer transformer / RMSNorm-MLP stacks. They render and run correctly; only the final-tile error exceeds the threshold.

## Setup

```sh
./scripts/setup.sh
```

Clones tinygrad into `./tinygrad/`, checks out the pin, applies the 3-line patch, and symlinks `tt_renderer.py` and `tt_device.py` into the tinygrad tree.

## Running tests

On a host with `ttl` / `ttnn` installed (e.g. inside the `tt-lang-dist` docker image, or directly on QuietBox):

```sh
python tt_tests.py          # 125-case end-to-end
python tt_runtime_smoke.py  # 36-case smoke
```
