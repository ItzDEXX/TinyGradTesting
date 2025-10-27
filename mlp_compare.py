import time
import numpy as np
from tinygrad import Tensor, TinyJit

np.set_printoptions(precision=6, suppress=True, linewidth=200)

WARMUP = 5
REPEAT = 30

Z  = np.load("params_dump.npz")
x  = Tensor(Z["x"])      # (B,L,D)
Wq = Tensor(Z["Wq"])     # (D,D)
Wk = Tensor(Z["Wk"])     # (D,D)
Wv = Tensor(Z["Wv"])     # (D,D)
W1 = Tensor(Z["W1"])     # (D,H)
b1 = Tensor(Z["b1"])     # (H,)
W2 = Tensor(Z["W2"])     # (H,D)
b2 = Tensor(Z["b2"])     # (D,)

def attn_single_head(x, Wq, Wk, Wv):
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    B, L, D = x.shape
    KT = K.transpose(-1, -2)                
    scores = (Q @ KT) * (1.0 / np.sqrt(D))   
    A = scores.softmax(axis=-1)
    return A @ V                             

@TinyJit
def attn_single_head_jit(x, Wq, Wk, Wv):
    return attn_single_head(x, Wq, Wk, Wv)


def mlp_baseline(h, W1, b1, W2, b2):
    y = h @ W1
    y = y + b1
    y = y.gelu()
    y = y @ W2
    y = y + b2
    return y

@TinyJit
def mlp_fused(h, W1, b1, W2, b2):
    y = (h @ W1 + b1).gelu()
    y = y @ W2 + b2
    return y

def bench(call, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup): _ = call().numpy()
    ts=[]
    for _ in range(repeat):
        t0=time.perf_counter(); _=call().numpy()
        ts.append((time.perf_counter()-t0)*1000.0)
    return float(np.median(ts))

h = attn_single_head(x, Wq, Wk, Wv)

def call_mlp_baseline():
    return mlp_baseline(h, W1, b1, W2, b2)

def call_mlp_fused():
    return mlp_fused(h, W1, b1, W2, b2)

p50_mlp_base = bench(call_mlp_baseline)
p50_mlp_fuse = bench(call_mlp_fused)
spd_mlp = 100.0 * (p50_mlp_base - p50_mlp_fuse) / max(p50_mlp_base, 1e-9)
print(f"[MLP-ONLY] baseline p50={p50_mlp_base:.3f} ms | fused p50={p50_mlp_fuse:.3f} ms | speedup={spd_mlp:.1f}%")
