# tinygrad-tt

A Tenstorrent backend for [tinygrad](https://github.com/tinygrad/tinygrad), targeting [TT-Lang](https://github.com/tenstorrent/tt-lang) as the codegen layer instead of raw tt-metal.

Status: PoC. On QuietBox with the latest `tt-lang-dist` image checked on 2026-05-15 (`ttl` 1.0.6), the staged hardware driver reports `pass=112 xfail=13 fail=0 err=0`. The renderer targets the current runnable API (`ttl.math.*`); the upstream spec appendix mentions `ttl.block.*`, but the current Python package and dist image do not expose `ttl.block`.

## What works

- **Elementwise**: relu, sigmoid, gelu, silu, tanh, sin, cos, exp, log, sqrt, rsqrt, square, neg, recip, sign-derived abs, scalar add/mul, div, maximum, minimum, squared_diff, fma (`a*b+c`), four_input (`a*b+c*d`)
- **Matmul**: square (64/128/256), rectangular, long-K, thin-M, transposed (`a @ b.T`, `a.T @ b`, including rectangular `a.T @ b`), with `+relu` / `+silu` / `+add` / `+bias` / `+scale` fusion
- **Attention**: `softmax((Q @ K.T) / √d, axis=1)` (4-kernel chain) and `softmax((Q @ K.T) / √d) @ V` (5-kernel chain — full scaled-dot-product attention), both at 64×64 and 128×128, plus full **attention block with residual** `x + softmax((Q @ K.T) / √d) @ V` (residual + into SV matmul epilogue, 5 kernels)
- **Generic elementwise chains**: e.g. `relu((a + b) * 0.5)` lowers as 3 separate elementwise kernels with auto-derived dataflow
- **Reductions**: sum / max / min / mean on axis=0 and axis=1, plus full-tensor sum/max/mean (no axis → scalar), including `(a-b).sum()` fused
- **Broadcast**: row, column, scalar
- **Non-tile-aligned shapes**: e.g. (48,48), (50,70), (50,40)@(40,70), softmax/layernorm at (96,96) — handled via ttnn TILE_LAYOUT padding + ceil-div tile loops
- **Multi-kernel chains**: sub_max, softmax (manual + native, ax=0 and ax=1), layernorm (manual + native + affine `*w+b`), var/std on axis=1, mlp_block
- **Native fused-reduce-with-prologue-and-epilogue**: `Tensor.softmax(axis=1)`, `Tensor.layernorm()`, and `rms_norm = x * rsqrt(mean(x²)+ε)` lower into multi-kernel chains and validate.
- **Transformer-adjacent chains that validate**: scaled-dot-product attention, attention with residual, cross-attention with rectangular Q/KV shapes, SwiGLU, SiLU MLP, masked attention logits, residual MLP blocks, `(a@b)@c`, L2 norm, mish, hardswish, positional add, and cosine similarity.

## Runtime boundary

`TT_DRYRUN=1` is the supported path today: tinygrad renders TT-Lang source, `tt_runner.py` stages importable kernels plus a manifest, and `tt_kernels_driver.py` runs those kernels on a TT host against torch references. The in-process `TT_DRYRUN=0` backend is still a skeleton; the allocator does not yet materialize `ttnn` tensors from tinygrad buffers.

## What doesn't yet work

- argmax / argmin — blocked at the ttl primitive layer: argmax lowering needs elementwise comparison (`eq`/`where`) and an iota/arange tile generator, neither of which are exposed by the hardware-mode `ttl.math` package (only `where` exists in the simulator). Not renderer-fixable without upstream ttl primitives.
- `Tensor.rand(device="TT")` cannot be lowered (threefry RNG uses bitwise ops the tile renderer doesn't handle); `tt_runner.py` uses explicit compile-only `tt_input(...)` placeholders, and the hardware driver fills real inputs from deterministic torch random values.
- Runtime backend V0 is bf16-on-tile, fp32-in-buffer, 2D-only. Higher-rank tensors and dtype control aren't plumbed.
- The validation driver marks 13 known numerical-drift cases as XFAIL under deterministic signed inputs and strict tolerances: `log_softmax`, `mlp_4x`, `geglu_ffn`, `decoder_block`, and the longer transformer/RMSNorm-MLP stacks listed in `EXPECTED_FAILURES`.

## Chain dataflow is auto-derived

Multi-kernel chains (softmax, layernorm, mlp_block, etc.) used to need hand-written wiring in the driver. Now the renderer emits a structured `# tt_contract: {...}` JSON contract in every kernel, the device records each DRYRUN call as `{name, src, buf_ids, contract}`, and `tt_runner.py` writes a `_manifest.json` pairing each kernel file with its kind, slot mapping, per-slot buf ids, and attributes such as matmul transpose flags. The driver walks each label's steps in order, threads ttnn tensors via buffer identity, and derives intermediate shapes from metadata rather than from generated function names.

## Repo layout

```
.
├── tt_renderer.py            # TT-Lang renderer (placed at tinygrad/tinygrad/renderer/tt.py)
├── tt_device.py              # TT device backend (placed at tinygrad/tinygrad/runtime/ops_tt.py)
├── tt_runner.py              # Renders all test cases as .py files under tt_kernels/
├── tt_kernels_driver.py      # Runs the staged kernels on a TT host and compares to torch
├── patches/
│   └── 0001-tt-device-hooks.patch   # 3 trivial upstream hooks (ALL_DEVICES, Renderer flag)
├── scripts/
│   └── setup.sh              # Clones tinygrad at the pin, applies the patch, links files
└── TINYGRAD_PIN              # Upstream tinygrad commit this PoC tracks
```

## Setup

```sh
./scripts/setup.sh
```

This clones tinygrad into `./tinygrad/`, checks out the pinned commit, applies the three-line hook patch, and symlinks `tt_renderer.py` and `tt_device.py` into the tinygrad tree.

## Generating kernels

```sh
python tt_runner.py
```

This writes one `.py` per kernel under `tt_kernels/`. `TT_DRYRUN=1` is set automatically so no TT hardware is required for this step.

## Running on a TT host

```sh
rsync -a tt_kernels tt-host:/path/to/staging/
rsync -a tt_kernels_driver.py tt-host:/path/to/staging/
ssh tt-host 'cd /path/to/staging && python tt_kernels_driver.py'
```

The driver expects `ttl` and `ttnn` to be importable (inside the `tt-lang-dist` docker image, or any host where the TT-Lang SDK is installed). For each test case it allocates deterministic torch inputs, reshapes 1-D logical inputs to their 2-D tile-device representation where needed, invokes the staged kernel(s), and compares against a torch reference.

## How it works (high level)

1. **`TTRenderer`** (`tt_renderer.py`) is a subclass of `tinygrad.renderer.Renderer`. Its `render(uops)` classifies each kernel as matmul, reduce, or elementwise and emits a TT-Lang Python source string.
2. **`TTDevice`** (`tt_device.py`) is a `Compiled` device. In `TT_DRYRUN` mode it records stable buffer ids, rendered source, and parsed kernel contracts as structured staging records. In live mode it would import and execute the rendered module, but full buffer-to-ttnn materialization is not implemented yet.
3. **`tt_kernels_driver.py`** runs on the TT host. It reads the generated manifest, maps external and intermediate buffers by id, chains outputs through ttnn tensors, and exits nonzero for any unexpected failure.

## Tinygrad upstream hooks

Three tiny edits to tinygrad upstream make this work:

| File                                  | Change |
|---------------------------------------|--------|
| `tinygrad/device.py`                  | Add `"TT"` to `ALL_DEVICES` so `Device["TT"]` resolves |
| `tinygrad/renderer/__init__.py`       | Add `disable_hand_coded_opts: bool = False` to `Renderer` |
| `tinygrad/codegen/opt/postrange.py`   | Honor `disable_hand_coded_opts` (the tile-natural shapes we want would be wrecked by tinygrad's heuristic opt pass) |

See `patches/0001-tt-device-hooks.patch`.
