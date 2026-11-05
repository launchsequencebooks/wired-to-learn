# REINFORCE: learning a policy directly with the policy-gradient theorem.
import json, time
import numpy as np
import torch
from torch import nn
import gymnasium as gym

torch.manual_seed(0)
env = gym.make("CartPole-v1")
policy = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))   # logits for left/right
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
GAMMA, EPISODES = 0.99, 600
returns, t0 = [], time.perf_counter()

for ep in range(EPISODES):
    obs, _ = env.reset(seed=ep)
    log_probs, rewards, done = [], [], False
    while not done:
        dist = torch.distributions.Categorical(logits=policy(torch.tensor(obs)))
        a = dist.sample()                                      # act by sampling the policy
        log_probs.append(dist.log_prob(a))
        obs, r, term, trunc, _ = env.step(a.item())
        rewards.append(r); done = term or trunc
    G, discounted = 0.0, []
    for r in reversed(rewards):                                # return from each step onward
        G = r + GAMMA * G
        discounted.insert(0, G)
    G = torch.tensor(discounted)
    G = (G - G.mean()) / (G.std() + 1e-8)                      # baseline: center and scale
    loss = -(torch.stack(log_probs) * G).sum()                 # make good actions more likely
    opt.zero_grad(); loss.backward(); opt.step()
    returns.append(sum(rewards))
    if (ep + 1) % 100 == 0:
        print(f"episodes {ep - 98:3d}-{ep + 1:3d}: average return {np.mean(returns[-100:]):6.1f}  "
              f"({time.perf_counter() - t0:.0f}s)")
json.dump({"returns": returns}, open("reinforce_returns.json", "w"))
