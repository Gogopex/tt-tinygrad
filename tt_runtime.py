"""User-facing helpers for in-process execution on the TT runtime backend.

`from_torch(t)` lifts a torch.Tensor into a tinygrad Tensor on TT device, and
registers the 2D shape on the underlying TTBuffer (tinygrad's Buffer API
only carries byte count, not shape). The runtime backend reads that shape
when materializing ttnn tensors at kernel-call time.

`to_torch(t)` realizes a TT-device tinygrad Tensor and returns a torch.Tensor.
"""
from __future__ import annotations
import torch
from tinygrad import Tensor

_SUPPORTED_TORCH_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def from_torch(t: torch.Tensor) -> Tensor:
  if t.ndim != 2: raise ValueError(f"TT runtime V0 expects 2D tensors, got shape {tuple(t.shape)}")
  if t.dtype not in _SUPPORTED_TORCH_DTYPES:
    raise ValueError(f"unsupported torch dtype {t.dtype}; supported: {sorted(d.__repr__() for d in _SUPPORTED_TORCH_DTYPES)}")
  t32 = t.to(torch.float32).contiguous()
  out = Tensor(t32.numpy(), device="TT").realize()
  buf = out.uop.buffer
  if not hasattr(buf, "_buf") or buf._buf is None:
    raise RuntimeError("expected realized Buffer to have ._buf TTBuffer attached")
  buf._buf.shape = tuple(t.shape)
  buf._buf.dtype_bytes = 4
  return out


def to_torch(t: Tensor) -> torch.Tensor:
  realized = t.realize()
  buf = realized.uop.buffer
  ttbuf = buf._buf
  if ttbuf.shape is None:
    raise RuntimeError("realized TT tensor has no registered shape; this is a runtime-backend bug")
  if getattr(ttbuf, "_dirty_bytes", False) and ttbuf.ttnn_tensor is not None:
    from tinygrad.runtime.ops_tt import _sync_ttnn_to_bytes
    _sync_ttnn_to_bytes(ttbuf)
  raw = bytes(ttbuf.mv)
  shape = ttbuf.shape
  n = shape[0] * shape[1]
  host_dtype = torch.float32 if ttbuf.dtype_bytes == 4 else torch.bfloat16
  arr = torch.frombuffer(bytearray(raw), dtype=host_dtype, count=n).reshape(shape).clone()
  return arr
