# What each head of the trained mini-GPT attends to: the statistics quoted in
# Section 18.5 for all 16 heads, and Figure 18.5 for the three heads named there.
# Run mini_gpt.py first (it saves mini_gpt.pt in this folder).
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1] / "scripts"))
import torch
from style import plt, save                       # the book's figure style
from mini_gpt import MiniGPT, chars, encode

LINE = "ROMEO:\nBut soft, what light"
SHOWN = LINE.index("But")                         # the figure shows 20 characters
FIGURE_HEADS = [(0, 3), (3, 1), (3, 3)]           # (layer, head) 0-based: 1/4, 4/2, 4/4
TITLES = ["layer 1, head 4: previous character",
          "layer 4, head 2: spaces and punctuation",
          "layer 4, head 4: further back"]

model = MiniGPT(len(chars))
model.load_state_dict(torch.load(here / "mini_gpt.pt", map_location="cpu"))
model.eval()
with torch.no_grad():
    model(encode(LINE)[None])                     # fills blk.attn.last_weights
T = len(LINE)
pos = torch.arange(T)
punct = torch.tensor([not c.isalnum() for c in LINE])   # spaces, newline, ':' and ','
print(f"{int(punct.sum())} of {T} characters are whitespace or punctuation")
print("layer head  prev-char  punctuation  mean distance")
for L, blk in enumerate(model.blocks):
    W = blk.attn.last_weights[0]                  # (heads, T, T) for the one sequence
    for h in range(W.shape[0]):
        prev = W[h, 1:, :-1].diagonal().mean()               # weight on position i-1
        on_punct = W[h][:, punct].sum(1).mean()              # weight on punctuation
        dist = (W[h] * (pos[:, None] - pos[None, :])).sum(1).mean()
        print(f"{L + 1:5d} {h + 1:4d}  {prev:9.2f}  {on_punct:11.2f}  {dist:13.1f}")

labels = [c if c != " " else "_" for c in LINE[SHOWN:]]
n = len(labels)
fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.6))
for ax, (L, h), title in zip(axes.flat, FIGURE_HEADS, TITLES):
    W = model.blocks[L].attn.last_weights[0, h, SHOWN:, SHOWN:]   # 20 x 20 sub-block
    ax.imshow(W, cmap="Greys", vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(length=0); ax.grid(False)
    ax.set_xlabel("looked at", fontsize=7); ax.set_ylabel("looking", fontsize=7)
axes[0, 0].plot(range(n - 1), range(1, n), color="0.6", lw=0.6)   # sub-diagonal
axes[1, 1].axis("off")
fig.colorbar(axes[0, 0].images[0], ax=axes[1, 1], fraction=0.5,
             label="attention weight")
fig.tight_layout()
save(fig, "ch18_attention")                       # -> figures/ch18_attention.pdf
