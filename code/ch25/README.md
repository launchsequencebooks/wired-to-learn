# Chapter 25 · Fine-Tuning and Parameter-Efficient Tuning

*Part V: Learning Beyond Labels* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 25.1 | [`lora.py`](lora.py) | LoRA from scratch. | printed in the book |
| 25.2 | [`finetune_alice.py`](finetune_alice.py) | Training from scratch, full fine-tuning and LoRA on a new domain. | [`finetune_alice.log`](finetune_alice.log) |
| 25.3 | [`lora_real_model.py`](lora_real_model.py) | The same `add_lora` on a real pretrained model (SmolLM2-135M). Optional: needs `pip install transformers`. | printed in the book |

If you have not run `code/ch18/mini_gpt.py` yet, do it now; it takes about eight minutes and saves `mini_gpt.pt`, which this chapter loads.

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
