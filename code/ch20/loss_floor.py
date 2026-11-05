# The irreducible noise-prediction loss ("floor") of Listing 20.2, estimated with NumPy
# only: a Gaussian-mixture posterior over sampled spiral centres (jitter 0.04).
# Section 20.3.1 quotes the mean over t.
import numpy as np, math
rng = np.random.default_rng(0)
def centres(n):
    t = np.sqrt(rng.random(n)) * 3 * math.pi
    return np.stack([t*np.cos(t), t*np.sin(t)], 1) / 5
def floor(beta_max, K=3000, N=600, T=100):
    betas = np.linspace(1e-4, beta_max, T); abar = np.cumprod(1-betas)
    C = centres(K); s2 = 0.04**2
    out = []
    for t in range(T):
        a = abar[t]; var = a*s2 + (1-a)          # x_t | centre ~ N(sqrt(a) c, var I)
        x0 = centres(N) + 0.04*rng.standard_normal((N,2))
        eps = rng.standard_normal((N,2))
        xt = np.sqrt(a)*x0 + np.sqrt(1-a)*eps
        d2 = ((xt[:,None,:]-np.sqrt(a)*C[None,:,:])**2).sum(-1)
        logw = -d2/(2*var); logw -= logw.max(1, keepdims=True)
        w = np.exp(logw); w /= w.sum(1, keepdims=True)
        # E[x0 | x_t, c] = c + (a s2/var) (x_t - sqrt(a) c)/sqrt(a), averaged over c
        Ec = w @ C
        Ex0 = Ec + (a*s2/var)*(xt - np.sqrt(a)*Ec)/np.sqrt(a)
        Eeps = (xt - np.sqrt(a)*Ex0)/np.sqrt(1-a)
        out.append(((eps-Eeps)**2).mean())
    return np.array(out)
# (0.1, 200) is the schedule of Listing 20.2
for bm, T in ((0.05, 100), (0.15, 100), (0.1, 200)):
    f = floor(bm, T=T)
    q = [0, T // 10, 3 * T // 10, 6 * T // 10, T - 1]
    print(f"beta_max {bm}, T {T}: " + "  ".join(f"t={t} {f[t]:.2f}" for t in q)
          + f"  mean over t {f.mean():.3f}")
