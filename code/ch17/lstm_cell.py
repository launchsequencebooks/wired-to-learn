# The LSTM cell written out gate by gate, checked against nn.LSTMCell.
import torch
from torch import nn

torch.manual_seed(0)
n_in, n_hidden = 3, 4
cell = nn.LSTMCell(n_in, n_hidden)
x = torch.randn(2, n_in)                                  # a batch of 2
h, c = torch.zeros(2, n_hidden), torch.zeros(2, n_hidden)

z = x @ cell.weight_ih.T + cell.bias_ih + h @ cell.weight_hh.T + cell.bias_hh
i, f, g, o = z.chunk(4, dim=1)                            # PyTorch's gate order
i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
g = torch.tanh(g)
c_new = f * c + i * g                                     # forget some, write some
h_new = o * torch.tanh(c_new)                             # read out through the output gate

h_ref, c_ref = cell(x, (h, c))
print("hidden state matches:", torch.allclose(h_new, h_ref, atol=1e-6))
print("cell state matches:  ", torch.allclose(c_new, c_ref, atol=1e-6))
print("gate values for example 1:")
for name, gate in [("input i", i), ("forget f", f), ("output o", o)]:
    print(f"  {name:9s}{gate[0].detach().numpy().round(2)}")
print("LSTM parameters:", sum(p.numel() for p in cell.parameters()),
      "vs RNN cell:", sum(p.numel() for p in nn.RNNCell(n_in, n_hidden).parameters()))
