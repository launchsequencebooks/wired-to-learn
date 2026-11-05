# A recurrent network unrolled by hand, checked against nn.RNN.
import torch
from torch import nn

torch.manual_seed(0)
n_in, n_hidden, T = 3, 5, 4
rnn = nn.RNN(n_in, n_hidden, batch_first=True)          # tanh RNN
x = torch.randn(1, T, n_in)                               # one sequence of 4 time steps

W_xh, W_hh = rnn.weight_ih_l0, rnn.weight_hh_l0           # the SAME weights at every step
b = rnn.bias_ih_l0 + rnn.bias_hh_l0
h = torch.zeros(n_hidden)                                 # the memory starts empty
for t in range(T):
    h = torch.tanh(W_xh @ x[0, t] + W_hh @ h + b)         # new memory from input + old memory
    print(f"t={t}: h = {h.detach().numpy().round(3)}")

out, h_last = rnn(x)
print("matches nn.RNN:", torch.allclose(h, h_last[0, 0], atol=1e-6))
print("parameters:", sum(p.numel() for p in rnn.parameters()), "(independent of sequence length)")
