"""TinyNet: a small, readable deep-learning library built in Chapter 8."""
from .layers import Layer, Dense, ReLU, Sigmoid, Tanh, Dropout
from .losses import SoftmaxCrossEntropy, BinaryCrossEntropy, MSE
from .optim import SGD
from .model import Sequential
from .data import batches, train_test_split, Standardizer
from .gradcheck import gradient_check

__version__ = "1.0"
