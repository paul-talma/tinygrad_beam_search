from tinygrad import Tensor

a = Tensor(list(range(1000))).reshape(10, 100)
b = Tensor(list(range(2000))).reshape(100, 20)
c = Tensor(list(range(200))).reshape(20, 10)
out = a @ b @ c
print(out.numpy())
