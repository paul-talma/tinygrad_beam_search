from tinygrad import nn, Tensor

if __name__ == "__main__":
    t = Tensor.randn(1, 3, 128, 128)
    w = Tensor.randn(3, 3, 3, 3)
    t.conv2d(w).realize()
