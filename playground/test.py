from tinygrad import Tensor

if __name__ ==  "__main__":
    a = Tensor(list(range(1024 * 1024))).reshape(1024, 1024)
    b = Tensor(list(range(1024 * 1024))).reshape(1024, 1024)
    c = a @ b
    c.realize()
