from tinygrad import Tensor
import numpy as np

a=Tensor([1.0,2.0,3.0])
b=Tensor([4.0,5.0,6.0])
c=a+b
print(c.numpy())