# Check that everything the book needs is installed, and which accelerator PyTorch can use.
import importlib
import platform
import sys

print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
packages = {"numpy": "I-II", "matplotlib": "I-II", "sklearn": "I-II", "torch": "III-VII", "gymnasium": "V",
            "onnxruntime": "VI", "fastapi": "VI", "pandas": "VII", "streamlit": "VII", "plotly": "VII"}
missing = []
for name, parts in packages.items():
    try:
        version = importlib.import_module(name).__version__
        print(f"  ok       {name:12s} {version:14s} (Part{'s' if '-' in parts else ''} {parts})")
    except ImportError:
        missing.append(name)
        print(f"  MISSING  {name:12s} {'':14s} (Part{'s' if '-' in parts else ''} {parts})")

import torch
if torch.cuda.is_available():
    device = f"cuda ({torch.cuda.get_device_name(0)})"
elif torch.backends.mps.is_available():
    device = "mps (Apple GPU)"
else:
    device = f"cpu ({torch.get_num_threads()} threads)"
print("PyTorch will train on:", device)
print("all set" if not missing else f"install the missing packages: pip install {' '.join(missing)}")
