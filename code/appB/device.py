# Write code once, run it on whatever hardware is available.
import torch

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
model = torch.nn.Linear(1000, 10).to(device)              # move the parameters
x = torch.randn(64, 1000, device=device)                  # create data where the model lives
y = model(x)
print(f"device: {device}; output {tuple(y.shape)} on {y.device}")
print("back on the CPU as NumPy:", y.detach().cpu().numpy().shape)   # needed for printing or plotting
