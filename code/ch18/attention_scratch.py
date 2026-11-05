# Scaled dot-product attention by hand, on a three-word "sentence".
import torch
import torch.nn.functional as F

torch.manual_seed(0)
words = ["the", "cat", "sat"]
d = 4
X = torch.randn(3, d)                       # one embedding vector per word
W_q, W_k, W_v = (torch.randn(d, d) for _ in range(3))

Q, K, V = X @ W_q, X @ W_k, X @ W_v         # queries, keys, values
scores = Q @ K.T / d ** 0.5                 # how well each query matches each key
weights = torch.softmax(scores, dim=-1)     # each row sums to 1
out = weights @ V                           # weighted average of the values

print("attention weights (row = word looking, column = word looked at):")
print("        " + "".join(f"{w:>7s}" for w in words))
for w, row in zip(words, weights):
    print(f"{w:>7s} " + "".join(f"{p:7.2f}" for p in row.tolist()))

# causal (masked) attention: a word may only look at itself and earlier words
mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
causal = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
print("causal weights, last row:", causal[2].numpy().round(2), "| first row:", causal[0].numpy().round(2))

ref = F.scaled_dot_product_attention(Q[None], K[None], V[None])[0]
print("matches PyTorch:", torch.allclose(out, ref, atol=1e-6))
