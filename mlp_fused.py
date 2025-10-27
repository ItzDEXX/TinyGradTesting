import numpy as np
from tinygrad import Tensor

np.set_printoptions(precision=6, suppress=True, linewidth=200)

Z  = np.load("params_dump.npz")
x  = Tensor(Z["x"])     
W1 = Tensor(Z["W1"])    
b1 = Tensor(Z["b1"])    
W2 = Tensor(Z["W2"])    
b2 = Tensor(Z["b2"])    

y = (x @ W1 + b1).gelu()
y = y @ W2 + b2

Y = y.numpy()
B, L, D = Y.shape
print(f"[FUSED-MLP] y.shape = {Y.shape}")
print("[FUSED-MLP] sample Y[0,0,:min(8,D)] =", Y[0,0,:min(8, D)])
