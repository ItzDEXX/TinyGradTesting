from tinygrad import Tensor, Device& print("tinygrad device:", Device.DEFAULT)& a=Tensor([1.,2.,3.]); b=Tensor([4.,5.,6.])& print("a+b =", (a+b).numpy()) 
