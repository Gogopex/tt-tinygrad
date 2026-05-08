from __future__ import annotations
import os, importlib.util, tempfile, hashlib, functools, itertools, json, re, atexit
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.helpers import getenv
from tinygrad.renderer.tt import TTRenderer

TT_DRYRUN = getenv("TT_DRYRUN", 1)

_CONTRACT_RE = re.compile(r"^# tt_contract: (\{.*\})$", re.M)
_dryrun_calls: list[dict] = []
_next_buffer_uid = itertools.count()
_open_devices: list = []

def _parse_contract(src: str) -> dict | None:
  m = _CONTRACT_RE.search(src)
  if m is None: return None
  return json.loads(m.group(1))

def _try_import_ttl():
  try:
    import ttl, ttnn
    return ttl, ttnn
  except Exception as e:
    return None, e

class TTCompiler(Compiler):
  def compile(self, src: str) -> bytes: return src.encode()

class TTBuffer:
  def __init__(self, size: int, mv: memoryview, ttnn_tensor=None, shape: tuple[int, int] | None = None, dtype_bytes: int = 4):
    self.uid = next(_next_buffer_uid)
    self.size, self.mv, self.ttnn_tensor, self.shape, self.dtype_bytes = size, mv, ttnn_tensor, shape, dtype_bytes
    self._dirty_bytes = False

class TTAllocator(Allocator['TTDevice']):
  def _alloc(self, size: int, options) -> TTBuffer:
    return TTBuffer(size, memoryview(bytearray(size)))
  def _free(self, opaque: TTBuffer, options): pass
  def _copyin(self, dest: TTBuffer, src: memoryview):
    dest.mv[:] = src
    dest.ttnn_tensor = None
    dest.shape = None
    dest._dirty_bytes = False
  def _copyout(self, dest: memoryview, src: TTBuffer):
    if src._dirty_bytes: _sync_ttnn_to_bytes(src)
    dest[:] = src.mv

def _sync_ttnn_to_bytes(buf: TTBuffer) -> None:
  import torch, ttnn
  t = ttnn.to_torch(buf.ttnn_tensor)
  if buf.shape is not None:
    t = t[:buf.shape[0], :buf.shape[1]]
  t = t.to(torch.float32).contiguous()
  raw = t.view(torch.uint8).numpy().tobytes()
  buf.mv[:] = raw
  buf._dirty_bytes = False

def _derive_elementwise_shape(in_shapes: list[tuple[int, int]]) -> tuple[int, int]:
  out_r, out_c = in_shapes[0]
  for r, c in in_shapes[1:]:
    if r != out_r and r != 1 and out_r != 1:
      raise RuntimeError(f"incompatible elementwise row dims: {r} vs {out_r}")
    if c != out_c and c != 1 and out_c != 1:
      raise RuntimeError(f"incompatible elementwise col dims: {c} vs {out_c}")
    out_r, out_c = max(out_r, r), max(out_c, c)
  return out_r, out_c

def _derive_matmul_shape(in_shapes: list[tuple[int, int]], a_t: bool, b_t: bool) -> tuple[int, int]:
  a, b = in_shapes[0], in_shapes[1]
  m = a[1] if a_t else a[0]
  n = b[0] if b_t else b[1]
  return m, n

def _derive_out_shape(kind: str, in_shapes: list[tuple[int, int]], attrs: dict) -> tuple[int, int]:
  if kind == "matmul":
    a_t = attrs.get("a_transposed")
    b_t = attrs.get("b_transposed")
    if a_t is None or b_t is None:
      raise RuntimeError(f"matmul contract missing transpose attrs: {attrs!r}")
    return _derive_matmul_shape(in_shapes, bool(a_t), bool(b_t))
  if kind == "elementwise":  return _derive_elementwise_shape(in_shapes)
  if kind == "reduce_axis0": return (1, in_shapes[0][1])
  if kind == "reduce_axis1": return (in_shapes[0][0], 1)
  if kind == "full_reduce":  return (1, 1)
  raise ValueError(f"unknown kernel kind {kind!r}")

class TTProgram:
  def __init__(self, dev: 'TTDevice', name: str, lib: bytes, **kwargs):
    self.dev, self.name, self.src = dev, name, lib.decode()
    self._fxn = None
    self._contract = _parse_contract(self.src)

  def _compile_python(self):
    digest = hashlib.sha256(self.src.encode()).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"tt_kernel_{self.name}_{digest}.py")
    with open(path, "w") as f: f.write(self.src)
    spec = importlib.util.spec_from_file_location(f"tt_kernel_{digest}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fxn = getattr(mod, self.name, None)
    if fxn is None: raise RuntimeError(f"generated module has no callable named {self.name!r}")
    return fxn

  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kw):
    if TT_DRYRUN:
      _dryrun_calls.append({
        "name": self.name,
        "src": self.src,
        "buf_ids": [b.uid for b in bufs],
        "contract": self._contract,
      })
      if getenv("TT_PRINT_SRC", 0):
        print(f"\n=== TT kernel: {self.name} ===\n{self.src}\n=== /TT ===\n")
      return 0.0
    if self.dev.ttl is None:
      raise RuntimeError(f"ttl/ttnn not importable on this host: {self.dev.ttl_err!r}. "
                         f"Set TT_DRYRUN=1 or run on a host with tt-lang installed.")
    if self._fxn is None: self._fxn = self._compile_python()
    if self._contract is None:
      raise RuntimeError(f"kernel {self.name!r} has no tt_contract metadata; cannot derive shapes for live exec")
    kind = self._contract["kind"]
    slots = self._contract["slots"]
    attrs = self._contract.get("attrs", {})
    in_bufs = [bufs[s] for s in slots[:-1]]
    out_buf = bufs[slots[-1]]
    if any(b is out_buf for b in in_bufs):
      raise RuntimeError(f"kernel {self.name!r} aliases output buffer to one of its inputs; "
                         f"V0 runtime does not yet support in-place ops")
    in_shapes: list[tuple[int, int]] = []
    for b in in_bufs:
      if b.shape is None:
        raise RuntimeError(f"input buffer to kernel {self.name!r} has no registered shape; "
                           f"build TT inputs via tt_runtime.from_torch")
      in_shapes.append(b.shape)
    out_shape = _derive_out_shape(kind, in_shapes, attrs)
    self._materialize_ttnn(in_bufs, in_shapes)
    out_buf.shape = out_shape
    if out_buf.dtype_bytes == 4 and in_bufs:
      out_buf.dtype_bytes = max(b.dtype_bytes for b in in_bufs)
    out_buf.ttnn_tensor = None
    self._materialize_ttnn([out_buf], [out_shape], zero_init=True)
    self._fxn(*[b.ttnn_tensor for b in in_bufs], out_buf.ttnn_tensor)
    out_buf._dirty_bytes = True
    return 0.0

  def _materialize_ttnn(self, bufs, shapes, zero_init: bool = False) -> None:
    import torch, ttnn
    for b, sh in zip(bufs, shapes):
      if b.ttnn_tensor is not None: continue
      n_elems = sh[0] * sh[1]
      if zero_init:
        t = torch.zeros(sh, dtype=torch.bfloat16)
      else:
        host_dtype = torch.float32 if b.dtype_bytes == 4 else torch.bfloat16
        np_bytes = bytes(b.mv)
        t = torch.frombuffer(bytearray(np_bytes), dtype=host_dtype, count=n_elems).reshape(sh).to(torch.bfloat16)
      b.ttnn_tensor = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev.ttnn_device)

class TTDevice(Compiled):
  def __init__(self, device: str):
    self.ttl, ttnn_or_err = _try_import_ttl()
    self.ttl_err = ttnn_or_err if self.ttl is None else None
    if self.ttl is None and not TT_DRYRUN:
      raise RuntimeError(f"TT device requires ttl/ttnn (set TT_DRYRUN=1 to render-only): {self.ttl_err!r}")
    self.ttnn_device = None
    if self.ttl is not None and not TT_DRYRUN:
      import ttnn
      self.ttnn_device = ttnn.open_device(device_id=0)
      _open_devices.append(self)
    super().__init__(device, TTAllocator(self), [TTRenderer], functools.partial(TTProgram, self), arch="wormhole_b0")

  def synchronize(self):
    if self.ttnn_device is None: return
    import ttnn
    ttnn.synchronize_device(self.ttnn_device)

  def finalize(self):
    if self.ttnn_device is None: return
    import ttnn
    try: ttnn.close_device(self.ttnn_device)
    except Exception: pass
    self.ttnn_device = None
    if self in _open_devices: _open_devices.remove(self)

@atexit.register
def _close_all_tt_devices():
  for d in list(_open_devices): d.finalize()
