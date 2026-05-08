from __future__ import annotations
import json
from dataclasses import dataclass
from tinygrad.uop.ops import Ops, UOp, GroupOp
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.renderer import Renderer
from tinygrad.helpers import to_function_name

TILE_BINARY = {
  Ops.ADD: ("add", "+"),
  Ops.SUB: ("sub", "-"),
  Ops.MUL: ("mul", "*"),
  Ops.FDIV: ("div", "/"),
  Ops.CDIV: ("div", "/"),
  Ops.MAX: ("max", None),
}

TILE_UNARY = {
  Ops.EXP2: "exp2",
  Ops.LOG2: "log",
  Ops.SIN: "sin",
  Ops.SQRT: "sqrt",
  Ops.RECIPROCAL: "recip",
  Ops.NEG: "neg",
}

@dataclass
class KernelShape:
  inputs: list[UOp]
  output: UOp
  body: UOp
  out_dtype: object
  input_axes: list[set[int]] | None = None
  output_axes: set[int] | None = None

@dataclass
class MatmulShape:
  a: UOp
  b: UOp
  out: UOp
  m_range: UOp
  n_range: UOp
  k_range: UOp
  acc_reg: UOp
  tail: UOp
  extra_inputs: list[UOp]
  extra_input_axes: dict[UOp, set[int]]
  a_transposed: bool
  b_transposed: bool

@dataclass
class FullReduceShape:
  out: UOp
  op: str
  inputs: list[UOp]
  prologue: UOp
  epilogue: UOp
  acc_reg: UOp
  red_range: UOp

@dataclass
class ReduceShape:
  out: UOp
  axis: int
  op: str
  inputs: list[UOp]
  input_axes: list[set[int]]
  prologue: UOp
  epilogue: UOp
  acc_reg: UOp
  loop_range: UOp
  red_range: UOp

def _walk_to_op(u: UOp, target_op) -> UOp | None:
  while u.op is not target_op and u.src: u = u.src[0]
  return u if u.op is target_op else None

def _range_bound(r: UOp) -> int | None:
  if r.op is not Ops.RANGE or not r.src: return None
  s = r.src[0]
  if s.op is not Ops.CONST: return None
  try: return int(s.arg)
  except Exception: return None

def _chain_metadata(kind: str, input_params: list[UOp], output_param: UOp, **attrs) -> str:
  slots = [int(p.arg) for p in input_params] + [int(output_param.arg)]
  contract = {"version": 1, "kind": kind, "slots": slots, "attrs": attrs}
  return f"# tt_contract: {json.dumps(contract, sort_keys=True, separators=(',', ':'))}"

def _collect_load_params_and_axes(u: UOp, exclude: set[UOp]) -> tuple[list[UOp], dict[UOp, set[int]]]:
  seen: list[UOp] = []
  axes: dict[UOp, set[int]] = {}
  visited: set[int] = set()
  def walk(x: UOp) -> None:
    if id(x) in visited: return
    visited.add(id(x))
    if x.op is Ops.LOAD:
      p = _walk_to_op(x.src[0], Ops.PARAM)
      if p is not None and p not in exclude:
        if p not in seen:
          seen.append(p)
          axes[p] = set()
        axes[p] |= _input_axes(x)
    for s in x.src: walk(s)
  walk(u)
  return seen, axes

def _collect_load_params(u: UOp, exclude: set[UOp]) -> list[UOp]:
  return _collect_load_params_and_axes(u, exclude)[0]

def _classify_matmul(uops: list[UOp]) -> MatmulShape | None:
  ranges = [u for u in uops if u.op is Ops.RANGE]
  loops = [r for r in ranges if r.arg[1].name == "LOOP"]
  reduces = [r for r in ranges if r.arg[1].name == "REDUCE"]
  if len(loops) != 2 or len(reduces) != 1: return None
  k_range = reduces[0]
  acc_update_store = next((u for u in uops if u.op is Ops.STORE
                           and u.src[1].op is Ops.ADD
                           and any(s.op is Ops.LOAD for s in u.src[1].src)
                           and any(s.op is Ops.MUL for s in u.src[1].src)), None)
  if acc_update_store is None: return None
  acc_load, mul = (acc_update_store.src[1].src if acc_update_store.src[1].src[0].op is Ops.LOAD
                   else acc_update_store.src[1].src[::-1])
  if acc_load.op is not Ops.LOAD or mul.op is not Ops.MUL: return None
  if not all(s.op is Ops.LOAD for s in mul.src): return None
  a_buf = _walk_to_op(mul.src[0].src[0], Ops.PARAM)
  b_buf = _walk_to_op(mul.src[1].src[0], Ops.PARAM)
  if a_buf is None or b_buf is None: return None
  out_stores = [u for u in uops if u.op is Ops.STORE and u is not acc_update_store
                and _walk_to_op(u.src[0], Ops.PARAM) is not None
                and _walk_to_op(u.src[0], Ops.PARAM) not in (a_buf, b_buf)
                and not _is_zero_const(u.src[1])]
  if not out_stores: return None
  out_store = out_stores[0]
  out_buf = _walk_to_op(out_store.src[0], Ops.PARAM)
  if out_buf is None or out_buf in (a_buf, b_buf): return None
  acc_reg = _walk_to_op(acc_load.src[0], Ops.DEFINE_REG)
  if acc_reg is None: return None
  tail = out_store.src[1]
  extra_inputs, extra_input_axes = _collect_load_params_and_axes(tail, exclude={a_buf, b_buf, out_buf})
  m_range, n_range = loops[0], loops[1]
  a_idx = mul.src[0].src[0]
  b_idx = mul.src[1].src[0]
  a_k_stride = _extract_range_stride(a_idx, k_range.arg[0])
  b_k_stride = _extract_range_stride(b_idx, k_range.arg[0])
  if a_k_stride is None or b_k_stride is None: return None
  a_transposed = a_k_stride != 1
  b_transposed = b_k_stride == 1
  return MatmulShape(a=a_buf, b=b_buf, out=out_buf, m_range=m_range, n_range=n_range, k_range=k_range,
                     acc_reg=acc_reg, tail=tail, extra_inputs=extra_inputs,
                     extra_input_axes=extra_input_axes,
                     a_transposed=a_transposed, b_transposed=b_transposed)

def _extract_range_stride(u: UOp, range_id: int) -> int | None:
  visited: set[int] = set()
  def walk(x: UOp) -> int | None:
    if id(x) in visited: return None
    visited.add(id(x))
    if x.op is Ops.RANGE and x.arg[0] == range_id: return 1
    if x.op is Ops.MUL:
      a, b = x.src
      for r, s in ((a, b), (b, a)):
        if r.op is Ops.RANGE and r.arg[0] == range_id and s.op is Ops.CONST:
          v = _const_value(s)
          try: return int(v)
          except Exception: return None
    for sub in x.src:
      res = walk(sub)
      if res is not None: return res
    return None
  return walk(u)

def _references_define_reg(u: UOp, reg: UOp) -> bool:
  visited: set[int] = set()
  def walk(x: UOp) -> bool:
    if id(x) in visited: return False
    visited.add(id(x))
    if x.op is Ops.LOAD and _walk_to_op(x.src[0], Ops.DEFINE_REG) is reg: return True
    return any(walk(s) for s in x.src)
  return walk(u)

def _classify_full_reduce(uops: list[UOp]) -> FullReduceShape | None:
  ranges = [u for u in uops if u.op is Ops.RANGE]
  loops = [r for r in ranges if r.arg[1].name == "LOOP"]
  reduces = [r for r in ranges if r.arg[1].name == "REDUCE"]
  if len(loops) != 0 or len(reduces) != 1: return None
  red_r = reduces[0]

  acc_update_store = None
  prologue_uop = None
  acc_reg = None
  reduce_op = None
  for u in uops:
    if u.op is not Ops.STORE: continue
    rhs = u.src[1]
    if rhs.op not in (Ops.ADD, Ops.MAX): continue
    if len(rhs.src) != 2: continue
    a, b = rhs.src
    for cand, other in ((a, b), (b, a)):
      if cand.op is not Ops.LOAD: continue
      reg = _walk_to_op(cand.src[0], Ops.DEFINE_REG)
      if reg is None: continue
      if _references_define_reg(other, reg): continue
      acc_update_store = u
      prologue_uop = other
      acc_reg = reg
      reduce_op = "sum" if rhs.op is Ops.ADD else "max"
      break
    if acc_update_store is not None: break
  if acc_update_store is None: return None

  input_loads: dict[UOp, UOp] = {}
  inputs_seen: list[UOp] = []
  visited: set[int] = set()
  def collect(x: UOp) -> None:
    if id(x) in visited: return
    visited.add(id(x))
    if x.op is Ops.LOAD:
      p = _walk_to_op(x.src[0], Ops.PARAM)
      if p is not None and p not in input_loads:
        input_loads[p] = x
        inputs_seen.append(p)
      return
    for s in x.src: collect(s)
  collect(prologue_uop)
  if not inputs_seen: return None

  out_store = None
  for u in uops:
    if u.op is not Ops.STORE or u is acc_update_store: continue
    out_param = _walk_to_op(u.src[0], Ops.PARAM)
    if out_param is None or out_param in inputs_seen: continue
    if not _references_define_reg(u.src[1], acc_reg): continue
    out_store = u
    break
  if out_store is None: return None
  out_buf = _walk_to_op(out_store.src[0], Ops.PARAM)
  return FullReduceShape(out=out_buf, op=reduce_op, inputs=inputs_seen,
                          prologue=prologue_uop, epilogue=out_store.src[1],
                          acc_reg=acc_reg, red_range=red_r)

def _classify_reduce(uops: list[UOp]) -> ReduceShape | None:
  ranges = [u for u in uops if u.op is Ops.RANGE]
  loops = [r for r in ranges if r.arg[1].name == "LOOP"]
  reduces = [r for r in ranges if r.arg[1].name == "REDUCE"]
  if len(loops) != 1 or len(reduces) != 1: return None
  loop_r, red_r = loops[0], reduces[0]

  acc_update_store = None
  prologue_uop = None
  acc_reg = None
  reduce_op = None
  for u in uops:
    if u.op is not Ops.STORE: continue
    rhs = u.src[1]
    if rhs.op not in (Ops.ADD, Ops.MAX): continue
    if len(rhs.src) != 2: continue
    a, b = rhs.src
    for cand, other in ((a, b), (b, a)):
      if cand.op is not Ops.LOAD: continue
      reg = _walk_to_op(cand.src[0], Ops.DEFINE_REG)
      if reg is None: continue
      if _references_define_reg(other, reg): continue
      acc_update_store = u
      prologue_uop = other
      acc_reg = reg
      reduce_op = "sum" if rhs.op is Ops.ADD else "max"
      break
    if acc_update_store is not None: break
  if acc_update_store is None: return None

  input_loads: dict[UOp, UOp] = {}
  inputs_seen: list[UOp] = []
  visited: set[int] = set()
  def collect(x: UOp) -> None:
    if id(x) in visited: return
    visited.add(id(x))
    if x.op is Ops.LOAD:
      p = _walk_to_op(x.src[0], Ops.PARAM)
      if p is not None and p not in input_loads:
        input_loads[p] = x
        inputs_seen.append(p)
      return
    for s in x.src: collect(s)
  collect(prologue_uop)
  if not inputs_seen: return None

  out_store = None
  for u in uops:
    if u.op is not Ops.STORE: continue
    if u is acc_update_store: continue
    out_param = _walk_to_op(u.src[0], Ops.PARAM)
    if out_param is None or out_param in inputs_seen: continue
    if not _references_define_reg(u.src[1], acc_reg): continue
    out_store = u
    break
  if out_store is None: return None
  out_buf = _walk_to_op(out_store.src[0], Ops.PARAM)
  epilogue = out_store.src[1]

  loop_id = loop_r.arg[0]
  red_id = red_r.arg[0]
  strided_load = None
  for p in inputs_seen:
    axes = _input_axes(input_loads[p])
    if loop_id in axes and red_id in axes:
      strided_load = input_loads[p]
      break
  if strided_load is None: return None
  offset = strided_load.src[0].src[1]
  loop_stride = _extract_range_stride(offset, loop_id)
  red_stride = _extract_range_stride(offset, red_id)
  if loop_stride is None or red_stride is None: return None
  axis = 1 if loop_stride > red_stride else 0

  input_axes_list = [_input_axes(input_loads[p]) for p in inputs_seen]
  return ReduceShape(out=out_buf, axis=axis, op=reduce_op, inputs=inputs_seen,
                     input_axes=input_axes_list, prologue=prologue_uop, epilogue=epilogue,
                     acc_reg=acc_reg, loop_range=loop_r, red_range=red_r)

def _ranges_used(u: UOp, acc: set[int] | None = None) -> set[int]:
  if acc is None: acc = set()
  visited: set[int] = set()
  def walk(x: UOp):
    if id(x) in visited: return
    visited.add(id(x))
    if x.op is Ops.RANGE:
      acc.add(x.arg[0])
      return
    for s in x.src: walk(s)
  walk(u)
  return acc

def _input_axes(load: UOp) -> set[int]:
  if load.op is not Ops.LOAD: return set()
  idx = load.src[0]
  if idx.op is not Ops.INDEX: return set()
  return _ranges_used(idx.src[1])

def _classify(uops: list[UOp]) -> KernelShape | None:
  globals_ = [u for u in uops if u.op is Ops.PARAM]
  stores = [u for u in uops if u.op is Ops.STORE]
  if not stores or not globals_: return None
  store = stores[0]
  out_buf = store.src[0]
  while out_buf.op is not Ops.PARAM and out_buf.src: out_buf = out_buf.src[0]
  if out_buf.op is not Ops.PARAM: return None
  inputs = [g for g in globals_ if g is not out_buf]
  if not inputs: return None
  output_axes = _ranges_used(store.src[0].src[1]) if store.src[0].op is Ops.INDEX else set()
  input_axes: list[set[int]] = []
  for g in inputs:
    axes: set[int] = set()
    for u in uops:
      if u.op is Ops.LOAD and _walk_to_op(u.src[0], Ops.PARAM) is g:
        axes |= _input_axes(u)
    input_axes.append(axes)
  return KernelShape(inputs=inputs, output=out_buf, body=store.src[1], out_dtype=store.src[1].dtype,
                     input_axes=input_axes, output_axes=output_axes)

def _is_zero_const(u: UOp) -> bool:
  if u.op is not Ops.CONST: return False
  v = u.arg
  if hasattr(v, 'value'): v = v.value
  try: return float(v) == 0.0
  except Exception: return False

def _match_relu(u: UOp) -> UOp | None:
  if u.op is not Ops.WHERE: return None
  cond, t, f = u.src
  if cond.op is not Ops.CMPLT: return None
  lo, hi = cond.src
  if _is_zero_const(lo) and t is hi and _is_zero_const(f): return hi
  if _is_zero_const(hi) and t is lo and _is_zero_const(f): return lo
  return None

def _is_const_value(u: UOp, target: float) -> bool:
  if u.op is not Ops.CONST: return False
  v = u.arg
  if hasattr(v, 'value'): v = v.value
  try: return float(v) == target
  except Exception: return False

def _match_sign(u: UOp) -> UOp | None:
  if u.op is not Ops.WHERE: return None
  ne_cond, lt_branch, zero_f = u.src
  if not _is_zero_const(zero_f): return None
  if ne_cond.op is not Ops.CMPNE: return None
  x_a, ne_zero = ne_cond.src
  if not _is_zero_const(ne_zero): return None
  if lt_branch.op is not Ops.WHERE: return None
  lt_cond, neg_one, pos_one = lt_branch.src
  if not (_is_const_value(neg_one, -1.0) and _is_const_value(pos_one, 1.0)): return None
  if lt_cond.op is not Ops.CMPLT: return None
  x_b, lt_zero = lt_cond.src
  if not _is_zero_const(lt_zero): return None
  if x_a is not x_b: return None
  return x_a

def _match_abs(u: UOp) -> UOp | None:
  if u.op is not Ops.MUL: return None
  a, b = u.src
  for x, s in ((a, b), (b, a)):
    sign_arg = _match_sign(s)
    if sign_arg is not None and sign_arg is x: return x
  return None

def _is_const_approx(u: UOp, target: float, rtol: float = 1e-4) -> bool:
  if u.op is not Ops.CONST: return False
  v = u.arg
  if hasattr(v, 'value'): v = v.value
  try: return abs(float(v) - target) <= rtol * abs(target)
  except Exception: return False

def _match_div_one_plus_exp2(u: UOp):
  if u.op is Ops.MUL:
    a, b = u.src
    for num, denom_node in ((a, b), (b, a)):
      if denom_node.op is Ops.RECIPROCAL:
        denom = denom_node.src[0]
        res = _match_one_plus_exp2(denom)
        if res is not None: return num, res
  if u.op is Ops.FDIV:
    num, denom = u.src
    res = _match_one_plus_exp2(denom)
    if res is not None: return num, res
  return None

def _match_one_plus_exp2(u: UOp):
  if u.op is not Ops.ADD: return None
  a, b = u.src
  for one, exp2_node in ((a, b), (b, a)):
    if _is_const_value(one, 1.0) and exp2_node.op is Ops.EXP2:
      return exp2_node.src[0]
  return None

def _match_silu(u: UOp) -> UOp | None:
  res = _match_div_one_plus_exp2(u)
  if res is None: return None
  num, exp2_arg = res
  if exp2_arg.op is not Ops.MUL: return None
  a, b = exp2_arg.src
  for x_node, c_node in ((a, b), (b, a)):
    if _is_const_approx(c_node, -1.4426950408889634) and x_node is num:
      return num
  return None

def _match_gelu(u: UOp) -> UOp | None:
  res = _match_div_one_plus_exp2(u)
  if res is None: return None
  num, exp2_arg = res
  if exp2_arg.op is not Ops.MUL: return None
  a, b = exp2_arg.src
  inner, scale = None, None
  for x, y in ((a, b), (b, a)):
    if _is_const_approx(y, -2.302208198144325):
      inner, scale = x, y
      break
  if inner is None: return None
  if inner.op is not Ops.ADD: return None
  x1, cube_term = inner.src
  for cand_x, cand_cube in ((x1, cube_term), (cube_term, x1)):
    if cand_x is not num: continue
    if cand_cube.op is not Ops.MUL: continue
    ca, cb = cand_cube.src
    for c_node, mul_node in ((ca, cb), (cb, ca)):
      if not _is_const_approx(c_node, 0.044715): continue
      if mul_node.op is not Ops.MUL: continue
      ma, mb = mul_node.src
      for sq_node, x_outer in ((ma, mb), (mb, ma)):
        if x_outer is not num: continue
        if sq_node.op is not Ops.MUL: continue
        sa, sb = sq_node.src
        if sa is num and sb is num: return num
  return None

def _const_value(u: UOp):
  val = u.arg
  if hasattr(val, 'value'): val = val.value
  if u.dtype in (dtypes.float, dtypes.float32, dtypes.float16, dtypes.bfloat16): return float(val)
  return int(val) if isinstance(val, (int, bool)) else val

def _render_tile_expr(u: UOp, buf_names: dict[UOp, str], template_blk: str, depth: int = 0) -> str | None:
  if depth > 16: return None
  abs_arg = _match_abs(u)
  if abs_arg is not None:
    a = _render_tile_expr(abs_arg, buf_names, template_blk, depth+1)
    return None if a is None else f"ttl.math.abs({a})"
  relu_arg = _match_relu(u)
  if relu_arg is not None:
    a = _render_tile_expr(relu_arg, buf_names, template_blk, depth+1)
    return None if a is None else f"ttl.math.relu({a})"
  gelu_arg = _match_gelu(u)
  if gelu_arg is not None:
    a = _render_tile_expr(gelu_arg, buf_names, template_blk, depth+1)
    return None if a is None else f"ttl.math.gelu({a})"
  silu_arg = _match_silu(u)
  if silu_arg is not None:
    a = _render_tile_expr(silu_arg, buf_names, template_blk, depth+1)
    return None if a is None else f"ttl.math.silu({a})"
  if u.op is Ops.LOAD:
    bidx = u.src[0]
    buf = bidx
    while buf.op not in (Ops.PARAM, Ops.DEFINE_REG) and buf.src: buf = buf.src[0]
    name = buf_names.get(buf)
    if name is None: return None
    return f"{name}_blk"
  if u.op is Ops.CONST:
    return f"ttl.math.fill({template_blk}, {_const_value(u)!r})"
  if u.op in TILE_BINARY:
    a = _render_tile_expr(u.src[0], buf_names, template_blk, depth+1)
    b = _render_tile_expr(u.src[1], buf_names, template_blk, depth+1)
    if a is None or b is None: return None
    name, infix = TILE_BINARY[u.op]
    if infix is not None: return f"({a} {infix} {b})"
    return f"ttl.math.{name}({a}, {b})"
  if u.op in TILE_UNARY:
    a = _render_tile_expr(u.src[0], buf_names, template_blk, depth+1)
    if a is None: return None
    if u.op is Ops.LOG2:
      return f"(ttl.math.log({a}) * ttl.math.fill({template_blk}, 1.4426950408889634))"
    return f"ttl.math.{TILE_UNARY[u.op]}({a})"
  if u.op is Ops.CAST:
    return _render_tile_expr(u.src[0], buf_names, template_blk, depth+1)
  return None

class TTRenderer(Renderer):
  device = "TT"
  suffix = "TT"
  supports_float4 = False
  has_local = False
  has_threads = False
  has_shared = False
  global_max = (8, 8, 1)
  local_max = None
  shared_max = 0
  tensor_cores = []
  disable_hand_coded_opts = True
  code_for_op = {op: (lambda: None) for op in (*TILE_BINARY.keys(), *TILE_UNARY.keys(), Ops.WHERE, Ops.CMPLT, Ops.CMPNE)}

  def render(self, uops: list[UOp]) -> str:
    name = next((u.arg.function_name for u in uops if u.op is Ops.SINK and u.arg is not None), "tt_kernel")
    name = to_function_name(name)
    mm = _classify_matmul(uops)
    if mm is not None: return _render_matmul(name, mm)
    rs = _classify_reduce(uops)
    if rs is not None and rs.axis in (0, 1): return _render_reduce(name, rs)
    frs = _classify_full_reduce(uops)
    if frs is not None: return _render_full_reduce(name, frs)
    shape = _classify(uops)
    if shape is None:
      return _render_unsupported(name, "kernel structure not recognized as elementwise or matmul", uops)
    buf_names = {shape.output: "out"}
    for i, g in enumerate(shape.inputs):
      buf_names[g] = f"in{i}"
    template_blk = f"{buf_names[shape.inputs[0]]}_blk"
    expr = _render_tile_expr(shape.body, buf_names, template_blk)
    if expr is None:
      return _render_unsupported(name, "body has unsupported UOps", uops)
    arg_parts = [f"{buf_names[g]}: ttnn.Tensor" for g in shape.inputs] + ["out: ttnn.Tensor"]
    head = f"def {name}({', '.join(arg_parts)}) -> None:"
    in_dfbs = "\n".join(
      f"    {buf_names[g]}_dfb = ttl.make_dataflow_buffer_like({buf_names[g]}, shape=(1, 1), block_count=2)"
      for g in shape.inputs)
    out_dfb = f"    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)"
    in_waits = ", ".join(f"{buf_names[g]}_dfb.wait() as {buf_names[g]}_blk" for g in shape.inputs)
    in_reserves = ", ".join(f"{buf_names[g]}_dfb.reserve() as {buf_names[g]}_blk" for g in shape.inputs)
    in_copies = "\n".join(
      f"                            tx_{buf_names[g]} = ttl.copy({buf_names[g]}[mt * {buf_names[g]}_mt_mult, nt * {buf_names[g]}_nt_mult], {buf_names[g]}_blk)\n"
      f"                            tx_{buf_names[g]}.wait()" for g in shape.inputs)
    bcast_locals = "\n".join(
      f"    {buf_names[g]}_mt_mult = 1 if -(-{buf_names[g]}.shape[0] // 32) > 1 else 0\n"
      f"    {buf_names[g]}_nt_mult = 1 if -(-{buf_names[g]}.shape[1] // 32) > 1 else 0"
      for g in shape.inputs)
    out_axes = shape.output_axes or set()
    bcast_compute_lines = []
    for i, g in enumerate(shape.inputs):
      in_axes = (shape.input_axes or [set()]*len(shape.inputs))[i]
      missing = out_axes - in_axes
      n = buf_names[g]
      if not missing: continue
      if missing == out_axes:
        bcast_compute_lines.append(f"                            {n}_blk = ttl.math.broadcast({n}_blk, out_blk, dims=[0, 1])")
      elif 0 in missing:
        bcast_compute_lines.append(f"                            {n}_blk = ttl.math.broadcast({n}_blk, out_blk, dims=[0])")
      elif 1 in missing:
        bcast_compute_lines.append(f"                            {n}_blk = ttl.math.broadcast({n}_blk, out_blk, dims=[1])")
    bcast_compute = ("\n".join(bcast_compute_lines) + "\n") if bcast_compute_lines else ""
    chain_meta = _chain_metadata("elementwise", shape.inputs, shape.output)
    src = f"""# auto-generated TT-Lang kernel
{chain_meta}
import ttl
import ttnn

@ttl.operation(grid=\"auto\")
{head}
    row_tiles = -(-out.shape[0] // 32)
    col_tiles = -(-out.shape[1] // 32)
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)
{bcast_locals}

{in_dfbs}
{out_dfb}

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for lr in range(rows_per_node):
            mt = node_row * rows_per_node + lr
            if mt < row_tiles:
                for lc in range(cols_per_node):
                    nt = node_col * cols_per_node + lc
                    if nt < col_tiles:
                        with ({in_waits}, out_dfb.reserve() as out_blk):
{bcast_compute}                            out_blk.store({expr})

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for lr in range(rows_per_node):
            mt = node_row * rows_per_node + lr
            if mt < row_tiles:
                for lc in range(cols_per_node):
                    nt = node_col * cols_per_node + lc
                    if nt < col_tiles:
                        with ({in_reserves}):
{in_copies}

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for lr in range(rows_per_node):
            mt = node_row * rows_per_node + lr
            if mt < row_tiles:
                for lc in range(cols_per_node):
                    nt = node_col * cols_per_node + lc
                    if nt < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[mt, nt])
                            tx.wait()
"""
    return src

def _render_matmul(name: str, mm: MatmulShape) -> str:
  extra_names = {p: f"in{i}" for i, p in enumerate(mm.extra_inputs)}
  buf_names: dict[UOp, str] = {mm.acc_reg: "acc", **extra_names}
  tail_expr = _render_tile_expr(mm.tail, buf_names, "acc_blk")
  if tail_expr is None or tail_expr == "acc_blk":
    tail_store = "out_blk.store(acc_blk)"
  else:
    tail_store = f"out_blk.store({tail_expr})"

  extra_args = "".join(f", {n}: ttnn.Tensor" for n in extra_names.values())
  extra_dfbs = "".join(
    f"\n    {n}_dfb = ttl.make_dataflow_buffer_like({n}, shape=(1, 1), block_count=2)"
    for n in extra_names.values())
  extra_waits = "".join(f", {n}_dfb.wait() as {n}_blk" for n in extra_names.values())
  extra_reserves = ", ".join(f"{n}_dfb.reserve() as {n}_blk" for n in extra_names.values())
  m_axis = mm.m_range.arg[0]
  n_axis = mm.n_range.arg[0]
  extra_bcasts: list[str] = []
  extra_copy_lines: list[str] = []
  for p, n in extra_names.items():
    axes = mm.extra_input_axes.get(p, set())
    uses_m = m_axis in axes
    uses_n = n_axis in axes
    idx = f"[{'mt' if uses_m else '0'}, {'nt' if uses_n else '0'}]"
    extra_copy_lines.append(
      f"                            tx_{n} = ttl.copy({n}{idx}, {n}_blk)\n"
      f"                            tx_{n}.wait()")
    missing_dims = []
    if not uses_m: missing_dims.append(0)
    if not uses_n: missing_dims.append(1)
    if missing_dims:
      extra_bcasts.append(f"                            {n}_blk = ttl.math.broadcast({n}_blk, out_blk, dims={missing_dims})")
  extra_bcast_compute = ("\n".join(extra_bcasts) + "\n") if extra_bcasts else ""
  extra_post_kloop = ""
  if extra_names:
    extra_copies = "\n".join(extra_copy_lines)
    extra_post_kloop = f"""
                        with {extra_reserves}:
{extra_copies}"""

  chain_meta = _chain_metadata("matmul", [mm.a, mm.b, *mm.extra_inputs], mm.out,
                               a_transposed=mm.a_transposed, b_transposed=mm.b_transposed)
  a_idx = "a[kt, mt]" if mm.a_transposed else "a[mt, kt]"
  b_idx = "b[nt, kt]" if mm.b_transposed else "b[kt, nt]"
  k_tiles_src = "a.shape[0]" if mm.a_transposed else "a.shape[1]"
  a_expr = "ttl.math.transpose(a_blk)" if mm.a_transposed else "a_blk"
  b_expr = "ttl.math.transpose(b_blk)" if mm.b_transposed else "b_blk"

  return f"""# auto-generated TT-Lang matmul kernel
{chain_meta}
import ttl
import ttnn

@ttl.operation(grid=\"auto\")
def {name}(a: ttnn.Tensor, b: ttnn.Tensor{extra_args}, out: ttnn.Tensor) -> None:
    row_tiles = -(-out.shape[0] // 32)
    col_tiles = -(-out.shape[1] // 32)
    k_tiles = -(-{k_tiles_src} // 32)
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2){extra_dfbs}

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for lr in range(rows_per_node):
            mt = node_row * rows_per_node + lr
            if mt < row_tiles:
                for lc in range(cols_per_node):
                    nt = node_col * cols_per_node + lc
                    if nt < col_tiles:
                        with acc_dfb.reserve() as acc_blk:
                            acc_blk.store(ttl.math.fill(acc_blk, 0))
                        for kt in range(k_tiles):
                            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, acc_dfb.wait() as pre_acc_blk:
                                with acc_dfb.reserve() as acc_blk:
                                    acc_blk.store(pre_acc_blk + {a_expr} @ {b_expr})
                        with acc_dfb.wait() as acc_blk{extra_waits}, out_dfb.reserve() as out_blk:
{extra_bcast_compute}                            {tail_store}

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for lr in range(rows_per_node):
            mt = node_row * rows_per_node + lr
            if mt < row_tiles:
                for lc in range(cols_per_node):
                    nt = node_col * cols_per_node + lc
                    if nt < col_tiles:
                        for kt in range(k_tiles):
                            with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                                tx_a = ttl.copy({a_idx}, a_blk)
                                tx_b = ttl.copy({b_idx}, b_blk)
                                tx_a.wait()
                                tx_b.wait(){extra_post_kloop}

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for lr in range(rows_per_node):
            mt = node_row * rows_per_node + lr
            if mt < row_tiles:
                for lc in range(cols_per_node):
                    nt = node_col * cols_per_node + lc
                    if nt < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[mt, nt])
                            tx.wait()
"""

def _reduce_classify_inputs(rs: ReduceShape) -> tuple[list[tuple[int, UOp]], list[tuple[int, UOp]]] | None:
  loop_id = rs.loop_range.arg[0]
  red_id = rs.red_range.arg[0]
  strided: list[tuple[int, UOp]] = []
  bcast_red: list[tuple[int, UOp]] = []
  for i, p in enumerate(rs.inputs):
    axes = rs.input_axes[i]
    if red_id in axes: strided.append((i, p))
    elif loop_id in axes: bcast_red.append((i, p))
    else: return None
  if not strided: return None
  return strided, bcast_red

def _reduce_buf_names(rs: ReduceShape, bcast_red: list[tuple[int, UOp]]) -> tuple[dict[UOp, str], dict[UOp, str], dict[UOp, str]]:
  base: dict[UOp, str] = {p: f"in{i}" for i, p in enumerate(rs.inputs)}
  pro_names = dict(base)
  for _, p in bcast_red: pro_names[p] = base[p] + "_bc"
  epi_names = dict(base)
  epi_names[rs.acc_reg] = "red"
  return base, pro_names, epi_names

def _render_reduce(name: str, rs: ReduceShape) -> str:
  ttl_op = f"ttl.math.reduce_{rs.op}"
  init_val = "0.0" if rs.op == "sum" else "-3.4028234663852886e+38"
  classified = _reduce_classify_inputs(rs)
  if classified is None: return _render_unsupported(name, f"reduce axis={rs.axis} inputs not handled")
  strided, bcast_red = classified
  base_names, pro_names, epi_names = _reduce_buf_names(rs, bcast_red)
  shape_ref = base_names[strided[0][1]]
  template_blk = f"{shape_ref}_blk"
  pro_expr = _render_tile_expr(rs.prologue, pro_names, template_blk)
  if pro_expr is None: return _render_unsupported(name, f"reduce axis={rs.axis} prologue not supported")
  epi_expr = _render_tile_expr(rs.epilogue, epi_names, "red_blk")
  if epi_expr is None: return _render_unsupported(name, f"reduce axis={rs.axis} epilogue not supported")
  acc_update = f"pre_acc + {pro_expr}" if rs.op == "sum" else f"ttl.math.max(pre_acc, {pro_expr})"

  if rs.axis == 0:
    shape_decls = (f"    row_tiles_in = -(-{shape_ref}.shape[0] // 32)\n"
                   f"    col_tiles_out = -(-out.shape[1] // 32)")
    node_div_decl = "cols_per_node = -(-col_tiles_out // grid_cols)"
    gate, iter_var, idx_var = "node_row == 0", "lc", "nt"
    iter_count, outer_bound = "cols_per_node", "col_tiles_out"
    outer_offset = "node_col * cols_per_node + lc"
    inner_iter, bcast_dim = "row_tiles_in", 0
    strided_idx, bcast_red_idx, out_idx = "[kt, nt]", "[0, nt]", "[0, nt]"
    t_dfb_decl = ""
    reduce_block = (f"                    with acc_dfb.wait() as acc, scaler_dfb.wait() as sc, red_dfb.reserve() as red_blk:\n"
                    f"                        red_blk.store({ttl_op}(acc, sc, dims=[0]))")
    out_store_stmt = "out_blk.store(epi_blk)"
  else:
    shape_decls = (f"    row_tiles_out = -(-out.shape[0] // 32)\n"
                   f"    col_tiles_in = -(-{shape_ref}.shape[1] // 32)")
    node_div_decl = "rows_per_node = -(-row_tiles_out // grid_rows)"
    gate, iter_var, idx_var = "node_col == 0", "lr", "mt"
    iter_count, outer_bound = "rows_per_node", "row_tiles_out"
    outer_offset = "node_row * rows_per_node + lr"
    inner_iter, bcast_dim = "col_tiles_in", 1
    strided_idx, bcast_red_idx, out_idx = "[mt, kt]", "[mt, 0]", "[mt, 0]"
    t_dfb_decl = "    t_dfb      = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)\n"
    reduce_block = (f"                    with acc_dfb.wait() as acc, t_dfb.reserve() as tt_blk:\n"
                    f"                        tt_blk.store(ttl.math.transpose(acc))\n"
                    f"                    with t_dfb.wait() as tv, scaler_dfb.wait() as sc, red_dfb.reserve() as red_blk:\n"
                    f"                        red_blk.store({ttl_op}(tv, sc, dims=[0]))")
    out_store_stmt = "out_blk.store(ttl.math.transpose(epi_blk))"

  in_args = ", ".join(f"{base_names[p]}: ttnn.Tensor" for p in rs.inputs)
  in_dfbs = "\n".join(
    f"    {base_names[p]}_dfb    = ttl.make_dataflow_buffer_like({base_names[p]}, shape=(1, 1), block_count=2)"
    for p in rs.inputs)
  in_waits = ", ".join(f"{base_names[p]}_dfb.wait() as {base_names[p]}_blk" for p in rs.inputs)
  bcast_lines = "\n".join(
    f"                                {base_names[p]}_bc_blk = ttl.math.broadcast({base_names[p]}_blk, {shape_ref}_blk, dims=[{bcast_dim}])"
    for _, p in bcast_red)
  bcast_prelude = (bcast_lines + "\n") if bcast_lines else ""
  strided_params = {p for _, p in strided}
  in_copies = "\n".join(
    f"                        with {base_names[p]}_dfb.reserve() as {base_names[p]}_blk:\n"
    f"                            tx = ttl.copy({base_names[p]}{strided_idx if p in strided_params else bcast_red_idx}, {base_names[p]}_blk)\n"
    f"                            tx.wait()"
    for p in rs.inputs)

  chain_meta = _chain_metadata(f"reduce_axis{rs.axis}", rs.inputs, rs.out)

  return f"""# auto-generated TT-Lang reduce_{rs.op} axis={rs.axis} kernel
{chain_meta}
import ttl
import ttnn

@ttl.operation(grid=\"auto\")
def {name}({in_args}, out: ttnn.Tensor) -> None:
{shape_decls}
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    {node_div_decl}

{in_dfbs}
    acc_dfb    = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
{t_dfb_decl}    scaler_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    red_dfb    = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    epi_dfb    = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out_dfb    = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        if {gate}:
            with scaler_dfb.reserve() as sc:
                sc.store(ttl.math.fill(sc, 1.0))
            for {iter_var} in range({iter_count}):
                {idx_var} = {outer_offset}
                if {idx_var} < {outer_bound}:
                    with acc_dfb.reserve() as acc:
                        acc.store(ttl.math.fill(acc, {init_val}))
                    for kt in range({inner_iter}):
                        with {in_waits}, acc_dfb.wait() as pre_acc:
                            with acc_dfb.reserve() as new_acc:
{bcast_prelude}                                new_acc.store({acc_update})
{reduce_block}
                    with red_dfb.wait() as red_blk, epi_dfb.reserve() as epi_blk:
                        epi_blk.store({epi_expr})
                    with epi_dfb.wait() as epi_blk, out_dfb.reserve() as out_blk:
                        {out_store_stmt}

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        if {gate}:
            for {iter_var} in range({iter_count}):
                {idx_var} = {outer_offset}
                if {idx_var} < {outer_bound}:
                    for kt in range({inner_iter}):
{in_copies}

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        if {gate}:
            for {iter_var} in range({iter_count}):
                {idx_var} = {outer_offset}
                if {idx_var} < {outer_bound}:
                    with out_dfb.wait() as out_blk:
                        tx = ttl.copy(out_blk, out{out_idx})
                        tx.wait()
"""

def _render_full_reduce(name: str, frs: FullReduceShape) -> str:
  ttl_op = f"ttl.math.reduce_{frs.op}"
  init_val = "0.0" if frs.op == "sum" else "-3.4028234663852886e+38"
  base_names = {p: f"in{i}" for i, p in enumerate(frs.inputs)}
  pro_names = dict(base_names)
  shape_ref = base_names[frs.inputs[0]]
  template_blk = f"{shape_ref}_blk"
  pro_expr = _render_tile_expr(frs.prologue, pro_names, template_blk)
  if pro_expr is None: return _render_unsupported(name, "full reduce prologue not supported")
  epi_names = {frs.acc_reg: "red2"}
  epi_expr = _render_tile_expr(frs.epilogue, epi_names, "red2_blk")
  if epi_expr is None: return _render_unsupported(name, "full reduce epilogue not supported")
  acc_update = f"pre_acc + {pro_expr}" if frs.op == "sum" else f"ttl.math.max(pre_acc, {pro_expr})"

  in_args = ", ".join(f"{base_names[p]}: ttnn.Tensor" for p in frs.inputs)
  in_dfbs = "\n".join(
    f"    {base_names[p]}_dfb    = ttl.make_dataflow_buffer_like({base_names[p]}, shape=(1, 1), block_count=2)"
    for p in frs.inputs)
  in_waits = ", ".join(f"{base_names[p]}_dfb.wait() as {base_names[p]}_blk" for p in frs.inputs)
  in_copies = "\n".join(
    f"                        with {base_names[p]}_dfb.reserve() as {base_names[p]}_blk:\n"
    f"                            tx = ttl.copy({base_names[p]}[mt, nt], {base_names[p]}_blk)\n"
    f"                            tx.wait()"
    for p in frs.inputs)

  chain_meta = _chain_metadata("full_reduce", frs.inputs, frs.out)

  return f"""# auto-generated TT-Lang full_reduce_{frs.op} kernel
{chain_meta}
import ttl
import ttnn

@ttl.operation(grid=\"auto\")
def {name}({in_args}, out: ttnn.Tensor) -> None:
    row_tiles_in = -(-{shape_ref}.shape[0] // 32)
    col_tiles_in = -(-{shape_ref}.shape[1] // 32)

{in_dfbs}
    acc_dfb    = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    red1_dfb   = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    t_dfb      = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    red2_dfb   = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    epi_dfb    = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out_dfb    = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        if node_row == 0 and node_col == 0:
            with scaler_dfb.reserve() as sc:
                sc.store(ttl.math.fill(sc, 1.0))
            with scaler_dfb.reserve() as sc:
                sc.store(ttl.math.fill(sc, 1.0))
            with acc_dfb.reserve() as acc:
                acc.store(ttl.math.fill(acc, {init_val}))
            for mt in range(row_tiles_in):
                for nt in range(col_tiles_in):
                    with {in_waits}, acc_dfb.wait() as pre_acc:
                        with acc_dfb.reserve() as new_acc:
                            new_acc.store({acc_update})
            with acc_dfb.wait() as acc, scaler_dfb.wait() as sc, red1_dfb.reserve() as red1_blk:
                red1_blk.store({ttl_op}(acc, sc, dims=[0]))
            with red1_dfb.wait() as r1, t_dfb.reserve() as t_blk:
                t_blk.store(ttl.math.transpose(r1))
            with t_dfb.wait() as tv, scaler_dfb.wait() as sc, red2_dfb.reserve() as red2_blk:
                red2_blk.store({ttl_op}(tv, sc, dims=[0]))
            with red2_dfb.wait() as red2_blk, epi_dfb.reserve() as epi_blk:
                epi_blk.store({epi_expr})
            with epi_dfb.wait() as epi_blk, out_dfb.reserve() as out_blk:
                out_blk.store(epi_blk)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        if node_row == 0 and node_col == 0:
            for mt in range(row_tiles_in):
                for nt in range(col_tiles_in):
{in_copies}

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        if node_row == 0 and node_col == 0:
            with out_dfb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()
"""

def _render_unsupported(name: str, reason: str, uops: list[UOp] = ()) -> str:
  trace = "\n".join(f"#   {i:3d}: {u.op.name:20s} {getattr(u.dtype, 'name', '?'):>10s}  arg={u.arg!r}" for i, u in enumerate(uops))
  return "\n".join([
    f"# tinygrad TT renderer placeholder: {reason}",
    f"# kernel name: {name}",
    "# this file is not importable; it is filtered out by tt_runner.py before staging.",
    "# uops:" if uops else "",
    trace if uops else "",
    f"raise NotImplementedError({reason!r})",
  ])
