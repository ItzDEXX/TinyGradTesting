import numpy as np
from tinygrad import Tensor


Z = np.load("params_dump.npz")
x  = Tensor(Z["x"])    
W1 = Tensor(Z["W1"])  
b1 = Tensor(Z["b1"])  
W2 = Tensor(Z["W2"])   
b2 = Tensor(Z["b2"])   

y = x @ W1            
y = y + b1            
y = y.gelu()         
y = y @ W2           
y = y + b2            

Y = y.numpy()
B, L, D = Y.shape
np.set_printoptions(precision=6, suppress=True, linewidth=200)
print(f"[BASELINE] y.shape = {Y.shape}")
print("[BASELINE] sample Y[0,0,:min(8,D)] =", Y[0,0,:min(8, D)])'''