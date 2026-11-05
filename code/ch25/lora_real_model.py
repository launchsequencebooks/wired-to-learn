# The book's own add_lora on a real pretrained language model: SmolLM2-135M, a Llama-
# style Transformer whose attention projections are nn.Linear layers named q_proj,
# k_proj, v_proj and o_proj. Optional: needs `pip install transformers` (not part of the
# required stack) and an internet connection on first use (about 270 MB of weights).
import sys, time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wtlkit import load_text, set_seed
from lora import add_lora

NAME = "HuggingFaceTB/SmolLM2-135M"                        # Apache-2.0, 135M parameters
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME).float()
for p in model.parameters():
    p.requires_grad = False                                 # freeze the whole model ...
add_lora(model, r=8, targets=("q_proj", "v_proj"))         # ... then add the LoRA paths
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"parameters: {total:,} total, {trainable:,} trainable ({trainable / total:.2%})")


def write(prompt="Alice was beginning to get very tired", n=50):
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids
    torch.manual_seed(7)
    out = model.generate(ids, max_new_tokens=n, do_sample=True, temperature=0.7,
                         pad_token_id=tok.eos_token_id)
    return prompt + tok.decode(out[0][ids.shape[1]:])


print("before:", write())
data = tok(load_text("alice.txt"), return_tensors="pt").input_ids[0]   # subword tokens
BLOCK, BS, STEPS = 64, 4, 100
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
set_seed(0); model.train(); t0 = time.perf_counter()
for step in range(1, STEPS + 1):
    ix = torch.randint(len(data) - BLOCK, (BS,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    loss = model(x, labels=x).loss                         # targets shifted internally
    opt.zero_grad(); loss.backward(); opt.step()
    if step == 1 or step % 25 == 0:
        print(f"step {step:3d}  loss {loss.item():.3f}  "
              f"{time.perf_counter() - t0:.0f}s")
print(f"{STEPS} LoRA steps on a 135M-parameter model: {time.perf_counter() - t0:.0f}s")
print("after: ", write())
# With Hugging Face's peft the same thing is
#   peft.get_peft_model(model, LoraConfig(r=8, target_modules=["q_proj", "v_proj"]))
