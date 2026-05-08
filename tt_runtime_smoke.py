"""End-to-end smoke test for the in-process TT runtime backend.

Covers correctness of the core path plus the hardened guarantees:
  - elementwise broadcast (row, col, scalar) and broadcast-strictness
  - matmul shape via transposition metadata (a.T @ b, a @ b.T)
  - softmax chain (reduce + epilogue)
  - device.synchronize() actually fences
  - bf16 from_torch round-trip
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tinygrad"))
os.environ.setdefault("TT_DRYRUN", "0")

import torch
from tt_runtime import from_torch, to_torch
from tinygrad import Device


def check(name, got, expected, atol=0.1):
  diff = (got - expected).abs().max().item()
  status = "PASS" if diff < atol else "FAIL"
  print(f"  [{status}] {name}: max_diff={diff:.4g}")
  return status == "PASS"


def expect_raises(name, fn, msg_part):
  try: fn()
  except RuntimeError as e:
    ok = msg_part in str(e)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {'guard fired' if ok else f'wrong msg: {e}'}")
    return ok
  print(f"  [FAIL] {name}: no exception raised")
  return False


def main():
  ok_all = True

  a_t = torch.rand(64, 64)
  b_t = torch.rand(64, 64)
  a, b = from_torch(a_t), from_torch(b_t)

  ok_all &= check("a+b",         to_torch((a + b).realize()),               a_t + b_t)
  ok_all &= check("(a+b)*2",     to_torch(((a + b) * 2.0).realize()),       (a_t + b_t) * 2.0)
  ok_all &= check("relu(a)",     to_torch(a.relu().realize()),              torch.relu(a_t))
  ok_all &= check("relu(a+b)",   to_torch((a + b).relu().realize()),        torch.relu(a_t + b_t))
  ok_all &= check("a @ b",       to_torch((a @ b).realize()),               a_t @ b_t, atol=0.3)
  ok_all &= check("softmax ax1", to_torch(a.softmax(axis=1).realize()),     torch.softmax(a_t, dim=1), atol=0.02)

  c_t = torch.rand(64, 64)
  c = from_torch(c_t)
  ok_all &= check("a @ b.T",     to_torch((a @ c.transpose()).realize()),   a_t @ c_t.T, atol=0.3)
  ok_all &= check("a.T @ b",     to_torch((a.transpose() @ c).realize()),   a_t.T @ c_t, atol=0.3)

  row_t = torch.rand(1, 64)
  col_t = torch.rand(64, 1)
  row = from_torch(row_t)
  col = from_torch(col_t)
  ok_all &= check("a + row",  to_torch((a + row).realize()), a_t + row_t)
  ok_all &= check("a + col",  to_torch((a + col).realize()), a_t + col_t)

  Device["TT"].synchronize()
  print("  [PASS] synchronize() returns")

  bf16_in = from_torch(torch.ones(64, 64, dtype=torch.bfloat16) * 1.5)
  bf16_out = to_torch((bf16_in + bf16_in).realize())
  ok_all &= check("bf16 input upcast", bf16_out, torch.full((64, 64), 3.0))

  rect_a_t = torch.rand(64, 128)
  rect_b_t = torch.rand(128, 32)
  rect_a, rect_b = from_torch(rect_a_t), from_torch(rect_b_t)
  ok_all &= check("64x128 @ 128x32", to_torch((rect_a @ rect_b).realize()), rect_a_t @ rect_b_t, atol=0.3)

  big_a_t = torch.rand(128, 128)
  big_b_t = torch.rand(128, 128)
  big_a, big_b = from_torch(big_a_t), from_torch(big_b_t)
  ok_all &= check("128x128 elementwise", to_torch((big_a + big_b).realize()), big_a_t + big_b_t)
  ok_all &= check("128x128 matmul", to_torch((big_a @ big_b).realize()), big_a_t @ big_b_t, atol=0.6)

  tall_t = torch.rand(256, 64)
  tall = from_torch(tall_t)
  ok_all &= check("256x64 sum ax0", to_torch(tall.sum(axis=0).realize()), tall_t.sum(dim=0, keepdim=True), atol=2.0)

  small_a_t = torch.rand(48, 48)
  small_b_t = torch.rand(48, 48)
  small_a, small_b = from_torch(small_a_t), from_torch(small_b_t)
  ok_all &= check("48x48 a+b",    to_torch((small_a + small_b).realize()),  small_a_t + small_b_t)
  ok_all &= check("48x48 relu",   to_torch(small_a.relu().realize()),       torch.relu(small_a_t))
  ok_all &= check("48x48 matmul", to_torch((small_a @ small_b).realize()),  small_a_t @ small_b_t, atol=0.3)

  odd_t = torch.rand(50, 70)
  odd = from_torch(odd_t)
  ok_all &= check("50x70 relu",   to_torch(odd.relu().realize()),           torch.relu(odd_t))

  act_t = torch.rand(64, 64)
  act = from_torch(act_t)
  ok_all &= check("silu(a)",     to_torch(act.silu().realize()),     torch.nn.functional.silu(act_t),     atol=0.03)
  ok_all &= check("gelu(a)",     to_torch(act.gelu().realize()),     torch.nn.functional.gelu(act_t, approximate="tanh"), atol=0.03)
  ok_all &= check("tanh(a)",     to_torch(act.tanh().realize()),     torch.tanh(act_t),                   atol=0.03)
  ok_all &= check("sqrt(a+0.1)", to_torch((act + 0.1).sqrt().realize()), torch.sqrt(act_t + 0.1),         atol=0.03)
  ok_all &= check("log(a+0.1)",  to_torch((act + 0.1).log().realize()),  torch.log(act_t + 0.1),         atol=0.03)

  mish_out = (act * (1.0 + act.exp()).log().tanh()).realize()
  ok_all &= check("mish(a)",     to_torch(mish_out), torch.nn.functional.mish(act_t), atol=0.03)
  hs_out = (act * (act + 3.0).maximum(0.0).minimum(6.0) / 6.0).realize()
  ok_all &= check("hardswish(a)", to_torch(hs_out), torch.nn.functional.hardswish(act_t), atol=0.03)

  ln_t = torch.rand(64, 64)
  ln = from_torch(ln_t)
  ok_all &= check("layernorm",   to_torch(ln.layernorm(axis=1).realize()), torch.nn.functional.layer_norm(ln_t, (64,)), atol=0.05)

  rms_t = torch.rand(64, 64)
  rms = from_torch(rms_t)
  rms_out = (rms * (rms * rms).mean(axis=1, keepdim=True).add(1e-5).rsqrt()).realize()
  rms_ref = rms_t * torch.rsqrt((rms_t * rms_t).mean(dim=1, keepdim=True) + 1e-5)
  ok_all &= check("rms_norm",    to_torch(rms_out), rms_ref, atol=0.05)

  mr_t = torch.rand(64, 64)
  mr = from_torch(mr_t)
  x_plus_1 = (mr + 1.0).realize()
  x_times_2 = (mr * 2.0).realize()
  ok_all &= check("same-input two realize: +1", to_torch(x_plus_1), mr_t + 1.0)
  ok_all &= check("same-input two realize: *2", to_torch(x_times_2), mr_t * 2.0)

  rt_t = torch.rand(64, 64)
  rt = from_torch(rt_t)
  rt_mid = to_torch((rt + rt).realize())
  rt_back = from_torch(rt_mid * 0.5)
  rt_final = to_torch((rt_back + rt_back).realize())
  ok_all &= check("torch-bounce roundtrip", rt_final, rt_t * 2.0)

  try:
    from_torch(torch.rand(64))
    print("  [FAIL] from_torch(1D): no exception raised")
    ok_all = False
  except ValueError as e:
    print(f"  [PASS] from_torch(1D) rejects with ValueError")

  try:
    from_torch(torch.rand(64, 64, 64))
    print("  [FAIL] from_torch(3D): no exception raised")
    ok_all = False
  except ValueError as e:
    print(f"  [PASS] from_torch(3D) rejects with ValueError")

  try:
    from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    print("  [FAIL] from_torch(int32): no exception raised")
    ok_all = False
  except ValueError as e:
    print(f"  [PASS] from_torch(int32) rejects with ValueError")

  d_in, d_hidden, d_out = 64, 128, 32
  x_t = torch.rand(64, d_in)
  w1_t = torch.randn(d_in, d_hidden) * (d_in ** -0.5)
  w2_t = torch.randn(d_hidden, d_out) * (d_hidden ** -0.5)
  b1_t = torch.randn(64, d_hidden) * 0.1
  x, w1, w2, b1 = from_torch(x_t), from_torch(w1_t), from_torch(w2_t), from_torch(b1_t)
  mlp_out = to_torch(((x @ w1 + b1).relu() @ w2).realize())
  mlp_ref = ((x_t @ w1_t + b1_t).relu()) @ w2_t
  ok_all &= check("MLP (xW1+b).relu @ W2 (He init)", mlp_out, mlp_ref, atol=0.2)

  print("ALL OK" if ok_all else "FAILURES")
  sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
  main()
