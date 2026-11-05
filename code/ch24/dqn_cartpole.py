# Deep Q-learning (DQN) balances a pole on a cart.
import copy, json, random, time
from collections import deque
import numpy as np
import torch
from torch import nn
import gymnasium as gym

torch.manual_seed(0); random.seed(0); np.random.seed(0)
env = gym.make("CartPole-v1")
env.action_space.seed(0)                                   # random actions must be seeded too
n_obs, n_act = env.observation_space.shape[0], env.action_space.n


def q_net():
    return nn.Sequential(nn.Linear(n_obs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_act))


online, target = q_net(), q_net()
target.load_state_dict(online.state_dict())               # a slowly-updated copy for stable targets
opt = torch.optim.Adam(online.parameters(), lr=5e-4)
replay = deque(maxlen=50_000)                              # experience replay memory
GAMMA, BATCH, EPISODES = 0.99, 64, 300
returns, steps, t0 = [], 0, time.perf_counter()


def greedy_return(net, seed):                              # play one episode without exploring
    obs, _ = env.reset(seed=seed)
    done, total = False, 0.0
    while not done:
        with torch.no_grad():
            obs, r, term, trunc, _ = env.step(int(net(torch.tensor(obs)).argmax()))
        total += r; done = term or trunc
    return total


best, best_score = copy.deepcopy(online), -1.0

for ep in range(EPISODES):
    obs, _ = env.reset(seed=ep)
    done, total = False, 0.0
    while not done:
        eps = max(0.02, 1.0 - steps / 8000)                # explore a lot at first
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(online(torch.tensor(obs)).argmax())
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        replay.append((obs, a, r, obs2, float(terminated)))
        obs, total, steps = obs2, total + r, steps + 1
        if len(replay) >= 1000:                            # learn from random past experience
            batch = random.sample(replay, BATCH)
            s, a_, r_, s2, d = (torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*batch))
            q = online(s).gather(1, a_.long()[:, None]).squeeze(1)
            with torch.no_grad():                            # Bellman target from the frozen copy
                y = r_ + GAMMA * (1 - d) * target(s2).max(1).values
            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10)
            opt.step()
        if steps % 500 == 0:
            target.load_state_dict(online.state_dict())
    returns.append(total)
    if (ep + 1) % 10 == 0:                                 # checkpoint: keep the best policy so far
        score = np.mean([greedy_return(online, 5_000 + k) for k in range(5)])
        if score > best_score:
            best, best_score = copy.deepcopy(online), score
    if (ep + 1) % 50 == 0:
        print(f"episodes {ep - 48:3d}-{ep + 1:3d}: average return {np.mean(returns[-50:]):6.1f}  "
              f"(epsilon {eps:.2f}, best check {best_score:5.1f}, {time.perf_counter() - t0:.0f}s)")

# evaluate the best checkpoint on 20 fresh episodes (500 is the maximum)
scores = [greedy_return(best, 10_000 + k) for k in range(20)]
print(f"greedy policy over 20 test episodes: mean {np.mean(scores):.1f}, min {min(scores):.0f}, max {max(scores):.0f}")
json.dump({"returns": returns, "test": scores}, open("dqn_returns.json", "w"))
