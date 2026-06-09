import torch
from torch import Tensor

torch._logging.set_logs(graph_code=True)

def unfused_add():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a + 2
    c = b + 3
    return c

@torch.compile
def fused_add():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a + 2
    c = b + 3
    return c

unfused_add()
fused_add()
