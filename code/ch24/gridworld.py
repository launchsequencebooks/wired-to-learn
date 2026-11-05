# Tabular Q-learning in a small gridworld, from scratch.
import json
import numpy as np

GRID = ["S...#....",
        ".##.#.##.",
        "....#..#.",
        ".#.....#.",
        ".#.###.#G",
        "...X....."]          # S start, G goal (+10), X trap (-10), # wall; every step costs -0.1
ROWS, COLS = len(GRID), len(GRID[0])
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]           # up, down, left, right
ARROWS = "^v<>"


def step(state, a):
    r, c = state
    dr, dc = ACTIONS[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS) or GRID[nr][nc] == "#":
        nr, nc = r, c                                   # bumping into a wall: stay put
    cell = GRID[nr][nc]
    if cell == "G":
        return (nr, nc), 10.0, True
    if cell == "X":
        return (nr, nc), -10.0, True
    return (nr, nc), -0.1, False


def q_learning(episodes=2000, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS, COLS, 4))
    start = next((r, c) for r in range(ROWS) for c in range(COLS) if GRID[r][c] == "S")
    returns = []
    for ep in range(episodes):
        eps = max(eps_end, eps_start * (1 - ep / (0.6 * episodes)))   # explore less over time
        s, total = start, 0.0
        for _ in range(200):
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))   # epsilon-greedy
            s2, reward, done = step(s, a)
            target = reward if done else reward + gamma * Q[s2].max()          # Bellman target
            Q[s][a] += alpha * (target - Q[s][a])                              # move toward it
            s, total = s2, total + reward
            if done:
                break
        returns.append(total)
    return Q, returns, start


if __name__ == "__main__":
    Q, returns, start = q_learning()
    for k in [0, 499, 999, 1999]:
        window = returns[max(0, k - 99):k + 1]
        print(f"episode {k + 1:4d}: average return of last {len(window):3d} episodes {np.mean(window):6.2f}")
    print("\ngreedy policy (arrows), walls #, goal G, trap X:")
    for r in range(ROWS):
        print("  " + " ".join(GRID[r][c] if GRID[r][c] in "#GX" else ARROWS[int(np.argmax(Q[r, c]))]
                              for c in range(COLS)))
    s, path = start, [start]
    for _ in range(50):
        s, _, done = step(s, int(np.argmax(Q[s])))
        path.append(s)
        if done:
            break
    print(f"greedy path from S reaches {GRID[s[0]][s[1]]} in {len(path) - 1} steps")
    json.dump({"V": Q.max(2).tolist(), "policy": Q.argmax(2).tolist(), "grid": GRID,
               "returns": returns, "path": path}, open("gridworld.json", "w"))
