# tinygrad_single_head.py
import numpy as np
from tinygrad.tensor import Tensor

np.random.seed(0)
B, L, D = 1, 4, 8

x  = Tensor(np.random.randn(B, L, D).astype(np.float32))   # lazy tensor
Wq = Tensor(np.random.randn(D, D).astype(np.float32))
Wk = Tensor(np.random.randn(D, D).astype(np.float32))
Wv = Tensor(np.random.randn(D, D).astype(np.float32))

Q = x.matmul(Wq)                                           # [B,L,D]
K = x.matmul(Wk)                                           # [B,L,D]
V = x.matmul(Wv)                                           # [B,L,D]

scores = Q.matmul(K.transpose(0,2,1)) * (1/np.sqrt(D))     # [B,L,L]
A = scores.softmax(axis=-1)                                # [B,L,L]
out = A.matmul(V)                                          # [B,L,D]

print(out.numpy().shape)  # forces realization: (1, 4, 8)
