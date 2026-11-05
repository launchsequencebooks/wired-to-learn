"""wtlkit: the small, reusable PyTorch training kit built in Chapter 10
and extended through Parts III and IV."""
from .data import mnist_tensors, make_loaders
from .train import set_seed, count_params, train_one_epoch, evaluate, fit
from .datasets import (load_fashion, load_text, load_temperatures, load_cora,
                       load_banking77, load_hdfs, load_movielens, load_bike,
                       FASHION_CLASSES)
from .perf import benchmark, model_size_mb, accuracy
