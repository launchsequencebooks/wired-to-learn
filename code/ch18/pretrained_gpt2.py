# Load a real pretrained GPT-2 (124 million parameters) and generate from "ROMEO:".
# Optional: needs the `transformers` package (pip install transformers), which is not
# part of the book's required stack, and an internet connection on first use.
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained("gpt2")          # BPE, 50,257 tokens
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()   # 12 layers, width 768
cfg = model.config
print(f"layers {cfg.n_layer}, width {cfg.n_embd}, context {cfg.n_positions}, "
      f"vocabulary {cfg.vocab_size}, parameters {model.num_parameters():,}")
ids = tok("ROMEO:", return_tensors="pt").input_ids       # token ids, not characters
print("tokens:", tok.convert_ids_to_tokens(ids[0]))
torch.manual_seed(42)
out = model.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.8)
print(tok.decode(out[0]))
