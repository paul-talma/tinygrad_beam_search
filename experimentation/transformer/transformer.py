# imports

import numpy as np
from matplotlib import pyplot as plt
import tqdm

from tinygrad import Tensor, dtypes, nn

# constants
INPUT_PATH = "/Users/paultalma/projects/tinygrad/experimentation/transformer/input.txt"
BATCH_SIZE = 64
N_EPOCHS = 1
N_EMBED = 32
N_HEADS = 4
BLOCK_SIZE = 5
N_LAYERS = 1
DATA_CUTOFF = 4096


# tokenize, build data
class Tokenizer:
    def __init__(self):
        self.text = open(INPUT_PATH).read()
        self.chars = sorted(set(self.text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    def encode(self, s):
        return [self.stoi[c] for c in s]
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    def get_data(self):
        return np.array(self.encode(self.text[:DATA_CUTOFF]), dtype=np.int32)


# model
# attention head
class AttentionHead:
    def __init__(self, head_size, n_embed, block_size):
        self.d = head_size
        self.causal_mask = (Tensor.ones((block_size, block_size)).triu(diagonal=1) * -float('inf'))
        self.WQ = nn.Linear(n_embed, head_size, bias=False)
        self.WK = nn.Linear(n_embed, head_size, bias=False)
        self.WV = nn.Linear(n_embed, head_size, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        T = x.shape[1]
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        S = Q.matmul(K.transpose(dim0=-1, dim1=-2)) * 1/np.sqrt(self.d)
        S = S * self.causal_mask[:T, :T]
        T = S.softmax(axis=-1)
        return T.matmul(V)


class MultiHeadAttention:
    def __init__(self, n_heads, head_size, n_embed, block_size):
        self.n_heads = n_heads
        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size

        self.heads = [AttentionHead(head_size, n_embed, block_size) for _ in range(n_heads)]
        self.output_proj = nn.Linear(n_embed, n_embed, bias=True)

    def __call__(self, x: Tensor) ->Tensor:
        outputs = [head(x) for head in self.heads]
        return self.output_proj(Tensor.cat(*outputs, dim=-1))

class MLP:
    def __init__(self, n_embed):
        self.n_embed = n_embed
        self.n_hidden = 4*n_embed
        self.l1 = nn.Linear(n_embed, self.n_hidden)
        self.l2 = nn.Linear(self.n_hidden, n_embed)
        self.relu = lambda x: x.maximum(0)

    def __call__(self, x: Tensor) -> Tensor:
        return self.l2(self.relu(self.l1(x)))

class Block:
    def __init__(self, n_embed, n_heads, block_size):
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.block_size = block_size
        self.head_size = n_embed // n_heads
        self.mha = MultiHeadAttention(self.n_heads, self.head_size, self.n_embed, self.block_size)
        self.mlp = MLP(self.n_embed)
        self.ln0 = nn.LayerNorm(self.n_embed)
        self.ln1 = nn.LayerNorm(self.n_embed)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.mha(self.ln0(x))
        return  x + self.mlp(self.ln1(x))

class GPT:
    def __init__(self, vocab_size, n_embed, block_size, n_heads, n_layers):
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(vocab_size, n_embed)
        self.blocks = [Block(n_embed, n_heads, block_size) for _ in range(n_layers)]
        self.ln = nn.LayerNorm(n_embed)
        self.vocab_proj = nn.Linear(n_embed, vocab_size)

    def __call__(self, x: Tensor) -> Tensor:
        x_tokens = self.token_embedding(x)
        x_pos = self.positional_embedding(Tensor.arange(x.shape[1]))
        x = x_tokens + x_pos
        for block in self.blocks:
            x = block(x)
        return self.vocab_proj(self.ln(x))

class Trainer:
    def __init__(self, model: GPT, batch_size: int, n_epochs: int, tokenizer: Tokenizer, optim: str = "adam"):
        self.model = model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.tokenizer = tokenizer
        self.optim = nn.optim.Adam(nn.state.get_parameters(model)) if optim == "adam" else nn.optim.SGD(nn.state.get_parameters(model))
        self.block_size = model.block_size
        self.loss_history = []

    def train(self):
        data = self.tokenizer.get_data()
        for e in range(self.n_epochs):
            print(f"Epoch: {e + 1}")
            for i in tqdm.tqdm(range(np.ceil(len(data) // self.batch_size))):
                # draw (B, T) samples from data
                start_indices = np.random.randint(0, len(data) - self.model.block_size - 1, self.batch_size)
                inputs = Tensor([data[idx:idx + self.block_size] for idx in start_indices], dtype=dtypes.int32).reshape(self.batch_size, self.block_size)
                targets = Tensor([data[idx + 1 : idx + self.block_size + 1] for idx in start_indices], dtype=dtypes.int32).reshape(self.batch_size, self.block_size)

                with Tensor.train():
                    # query model
                    logits = self.model(inputs)

                    # compute loss (B*T, vocab size)
                    logits = logits.reshape(self.batch_size * self.block_size, -1)
                    targets = targets.reshape(self.batch_size * self.block_size)
                    loss = logits.cross_entropy(targets)
                    self.loss_history.append(loss.item())

                    # backward pass (zerograd, backward, step)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

        plt.plot(self.loss_history)
        plt.show()


if __name__ == "__main__":
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    model = GPT(vocab_size, N_EMBED, BLOCK_SIZE, N_HEADS, N_LAYERS)
    trainer = Trainer( model, BATCH_SIZE, N_EPOCHS, tokenizer)
    trainer.train()

    plt.plot(trainer.loss_history)

