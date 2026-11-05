# Retrieval-augmented generation with a small open instruct model, on the CPU.
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import SYSTEM, build_prompt
from qa import BookQA

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"                        # about 1 GB on first download
tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
lm.eval()


def generate(qa, question, max_new_tokens=150):
    """Retrieve, build the grounded prompt, and let the model answer from the passages only."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(qa, question)}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)["input_ids"]
    with torch.no_grad():
        out = lm.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    qa = BookQA()
    for q in ["What is the vanishing gradient problem?",
              "Why use a validation set?",
              "How do I bake sourdough bread?",
              "What is the capital of Australia?"]:
        t0 = time.perf_counter()
        answer = generate(qa, q)
        print(f"Q: {q}\nA: {answer}\n   ({time.perf_counter() - t0:.1f} s)\n")
