# Affine quantization from scratch: floats -> small integers -> floats.
import torch


def quantize(x, bits=8, per_channel=False):
    """Map x to integers 0..2^bits-1 with a scale and zero point.
    per_channel=True uses a separate range for each output channel (dim 0)."""
    if per_channel:
        dims = tuple(range(1, x.dim()))
        lo, hi = x.amin(dim=dims, keepdim=True), x.amax(dim=dims, keepdim=True)
    else:
        lo, hi = x.min(), x.max()
    lo, hi = torch.clamp(lo, max=0), torch.clamp(hi, min=0)   # make sure 0.0 is representable
    qmax = 2 ** bits - 1
    scale = (hi - lo).clamp(min=1e-12) / qmax
    zero_point = torch.round(-lo / scale)
    q = torch.clamp(torch.round(x / scale) + zero_point, 0, qmax)
    return q, scale, zero_point


def dequantize(q, scale, zero_point):
    return (q - zero_point) * scale


def fake_quantize(x, bits=8, per_channel=False):
    """Quantize and immediately dequantize: the float values an integer model would see."""
    return dequantize(*quantize(x, bits, per_channel))


if __name__ == "__main__":
    x = torch.tensor([-1.0, -0.4, 0.0, 0.25, 0.85, 2.0])
    q, s, z = quantize(x, bits=4)
    print("original:   ", [round(v, 3) for v in x.tolist()])
    print("4-bit codes:", q.int().tolist(), f" scale {s.item():.4f}, zero point {int(z)}")
    print("recovered:  ", [round(v, 3) for v in dequantize(q, s, z).tolist()])
