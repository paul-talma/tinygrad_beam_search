from tinygrad import Tensor

a = Tensor([[1, 2, 3], [4, 5, 6]])
b = a + 2
c = b + 3
b.realize()
c.realize()


# tg0 [x x x]
# tg1 [x x x]

# a
# [[1, 2, 3],
#  [4, 5, 6]]
# b
# [[7, 8, 9],
#  [0, 1, 2]]
# d
# [x, x]
