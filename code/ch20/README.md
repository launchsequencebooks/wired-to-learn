# Chapter 20 · Generative Models: GANs and Diffusion

*Part IV: The Network Zoo* · [All chapters](../../README.md#chapters)

Run every command from this folder, inside your activated environment.

| Listing | File | What it does | Expected output |
|---|---|---|---|
| 20.1 | [`gan_mnist.py`](gan_mnist.py) | A GAN that learns to draw digits. | [`gan_mnist.log`](gan_mnist.log) |
| 20.2 | [`diffusion_2d.py`](diffusion_2d.py) | A denoising diffusion model on a 2-D spiral. | [`diffusion_2d.log`](diffusion_2d.log) |

**Supporting file:** [`loss_floor.py`](loss_floor.py) (NumPy only) estimates the irreducible noise-prediction loss of Listing 20.2's spiral and schedule, the "floor" quoted in Section 20.3.1 (about 0.24 averaged over t).

Found a problem? [Report it](../../../../issues/new/choose) and include the listing number.
