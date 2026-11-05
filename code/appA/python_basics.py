# Python in one page: the parts of the language this book uses.
temperature = 21.5                       # a float; ints, strings and booleans work the same way
name = "neuron"
print(f"{name} at {temperature} degrees")   # f-strings put values inside text

layers = [784, 256, 128, 10]             # a list: ordered, changeable
layers.append(5)
print("layers:", layers, "first:", layers[0], "last two:", layers[-2:])

config = {"lr": 0.1, "epochs": 20}       # a dictionary: names -> values
config["batch_size"] = 64
for key, value in config.items():
    print(f"  {key} = {value}")

squares = [n * n for n in range(5)]      # a list comprehension builds a list in one line
print("squares:", squares)


def relu(x):
    """Functions take inputs and return outputs."""
    return x if x > 0 else 0.0


print("relu(-2) =", relu(-2), " relu(3) =", relu(3))


class Counter:
    """Classes bundle data (attributes) with the functions that use it (methods)."""

    def __init__(self):
        self.count = 0

    def step(self, n=1):
        self.count += n
        return self.count


c = Counter()
c.step(); c.step(5)
print("counter:", c.count)
