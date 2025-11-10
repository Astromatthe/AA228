import pandas as pd
import numpy as np
import time

def build_model(df, n_states, n_actions):
    # Build transition and reward matrices from data frame (for small dataset)
    counts = np.zeros((n_states, n_actions, n_states), dtype=int)   # keep track of counts of each s->a->sp transition
    sum_rewards = np.zeros((n_states, n_actions), dtype=float)  # collect rewards of state and performed action
    n_sa = np.zeros((n_states, n_actions), dtype=int)   # collect number of visits

    for _, row in df.iterrows():
        # extract state, action, and next state
        s = int(row['s']) - 1
        a = int(row['a']) - 1
        r = float(row['r'])
        sp = int(row['sp']) - 1

        # increase counts, rewards and visits
        counts[s,a,sp] += 1
        sum_rewards[s,a] += r
        n_sa[s,a] += 1

    # assemble matrices
    T = counts / counts.sum(axis = 2, keepdims=True)
    R = np.divide(sum_rewards, n_sa, out = np.zeros_like(sum_rewards), where=n_sa>0)    # average return of s,a combination

    return T,R

def value_iteration(P,R, gamma = 0.95, tol=1e-6, max_iter=10000):
    # Performs value iteration to compute optimal value function and policy
    # get shape of P
    n_states, n_actions, _ = np.shape(P)
    # initialize U
    U = np.zeros(n_states, dtype=float)

    for it in range(max_iter):
        Q = R + gamma * (P @ U) # lookahead equation
        U_new = np.max(Q, axis=1)   # value iteration maximizes U
        if np.max(np.abs(U_new - U)) < tol:
            break
        U = U_new

    policy = np.argmax(R + gamma * (P @ U), axis = 1)   # extract policy

    return U, policy

# START
df = pd.read_csv("data/small.csv")
mark = time.time()
T,R = build_model(df, n_states=100, n_actions=4)
print(f"Model built in {time.time() - mark:.2f}s")
mark = time.time()
U, policy = value_iteration(T,R,gamma = 0.95)
print(f"Value iteration completed in {time.time() - mark:.2f}s")

# save policy to file
np.savetxt("small.policy", policy + 1, fmt='%d')
policy += 1  # shift actions back to 1-4

# plot policy arrows in grid
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
X, Y = np.meshgrid(np.arange(10), np.arange(10))
U_arrows = np.zeros_like(X, dtype=float)
V_arrows = np.zeros_like(Y, dtype=float)
for s in range(100):
    a = policy[s]
    x = s % 10
    y = s // 10
    if a == 1:   # up
        U_arrows[y,x] = 0
        V_arrows[y,x] = 1
    elif a == 2: # right
        U_arrows[y,x] = 1
        V_arrows[y,x] = 0
    elif a == 3: # down
        U_arrows[y,x] = 0
        V_arrows[y,x] = -1
    elif a == 4: # left
        U_arrows[y,x] = -1
        V_arrows[y,x] = 0
ax = plt.gca()
# draw arrows; use scale_units='xy' and scale=1 so arrow lengths are in data coordinates
plt.quiver(X, Y, U_arrows, V_arrows, angles='xy', scale_units='xy', scale=1, pivot='middle')        

# set limits so grid lines fall on cell boundaries and arrows are centered in cells
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-0.5, 9.5)

# show integer tick labels at cell centers
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))

# create minor ticks at the cell boundaries (these will form the square grid)
ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)

# draw grid lines on minor ticks to show distinct squares
ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
# optionally hide the major grid
ax.grid(which='major', linestyle='')

# make cells square
ax.set_aspect('equal')

plt.title("Optimal Policy Arrows")
plt.show()