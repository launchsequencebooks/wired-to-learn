# Six optimizers, written from scratch in NumPy, racing on a narrow valley:
#   L(x, y) = 0.05 x^2 + 2 y^2   (gentle along x, steep along y)
import numpy as np

def loss(p):  return 0.05 * p[0]**2 + 2.0 * p[1]**2
def grad(p):  return np.array([0.1 * p[0], 4.0 * p[1]])


def run(name, lr, steps=100, start=(-9.0, 1.5), beta=0.9, beta2=0.999, eps=1e-8):
    p = np.array(start); v = np.zeros(2); s = np.zeros(2); path = [p.copy()]
    for t in range(1, steps + 1):
        g = grad(p)
        if name == "SGD":
            p = p - lr * g
        elif name == "Momentum":
            v = beta * v - lr * g; p = p + v
        elif name == "Nesterov":                       # look ahead, then correct
            g = grad(p + beta * v); v = beta * v - lr * g; p = p + v
        elif name == "AdaGrad":
            s = s + g**2; p = p - lr * g / (np.sqrt(s) + eps)
        elif name == "RMSProp":
            s = beta * s + (1 - beta) * g**2; p = p - lr * g / (np.sqrt(s) + eps)
        elif name == "Adam":
            v = beta * v + (1 - beta) * g; s = beta2 * s + (1 - beta2) * g**2
            v_hat, s_hat = v / (1 - beta**t), s / (1 - beta2**t)   # bias correction
            p = p - lr * v_hat / (np.sqrt(s_hat) + eps)
        path.append(p.copy())
    return np.array(path)


SETTINGS = {"SGD": 0.45, "Momentum": 0.05, "Nesterov": 0.05,
            "AdaGrad": 1.5, "RMSProp": 0.3, "Adam": 0.5}

if __name__ == "__main__":
    print(f"{'optimizer':10s} {'lr':>5s} {'loss@10':>9s} {'loss@50':>9s} {'loss@100':>9s}")
    for name, lr in SETTINGS.items():
        path = run(name, lr)
        L = [loss(path[k]) for k in (10, 50, 100)]
        print(f"{name:10s} {lr:5.2f} {L[0]:9.4f} {L[1]:9.4f} {L[2]:9.2e}")
