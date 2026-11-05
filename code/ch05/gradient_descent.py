# Gradient descent on a one-parameter "loss": L(w) = (w - 3)^2 + 1.
# The minimum is at w = 3. Try different learning rates.

def loss(w):
    return (w - 3) ** 2 + 1

def grad(w):
    return 2 * (w - 3)

for lr in [0.1, 0.5, 0.95, 1.05]:
    w = -2.0                         # start far from the minimum
    for step in range(20):
        w = w - lr * grad(w)         # the whole algorithm is this line
    print(f"lr={lr:<4}  after 20 steps: w={w:10.4f}  loss={loss(w):12.4f}")
