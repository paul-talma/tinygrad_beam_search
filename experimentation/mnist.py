import timeit

from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist


class Model:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))


X_train, y_train, X_test, y_test = mnist()
print(X_train.shape, X_train.dtype)

model = Model()
optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128

def step():
    Tensor.training = True
    samples = Tensor.randint(batch_size, high = X_train.shape[0])
    X, y = X_train[samples], y_train[samples]
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(y).backward()
    optim.step()
    return loss

jit_step = TinyJit(step)

print(timeit.repeat(step, repeat=5, number=1))
print(timeit.repeat(jit_step, repeat=5, number=1))

for s in range(7000):
    loss = jit_step()
    if s%100 == 0:
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == y_test).mean().item()
        print(f"step {s:4d}, loss {loss.item():.2f}, add {acc*100.:.2f}%")
