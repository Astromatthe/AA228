import numpy as np
import pandas as pd
from collections import defaultdict
import time
import bisect

# Hyperparameters
N_STATES = 302020
N_ACTIONS = 9
GAMMA = 0.95 # discount factor
ALPHA = 0.2 # learning rate for Q-learning
MAX_ITER = 1000
TOL = 1e-4
DEFAULT = 1

def build_model(df):
    T = defaultdict(lambda: defaultdict(list))
    for row in df.itertuples(index=False):
        T[row.s][row.a].append((row.r, row.sp)) 
    return T

def initialize_Q(T):
    Q = defaultdict(lambda: defaultdict(float))
    for s, acts in T.items():
        for a in acts:
            Q[s][a] = 0.0
    return Q

def fitted_q_iteration(T,Q, gamma = 0.95, max_iter = 200, tol = 1e-4):
    for it in range(1, max_iter+1):
        start = time.time()

        Q_new = defaultdict(lambda: defaultdict(float))
        delta = 0.0
        count = 0

        for s, acts in T.items(): 
            for a, samples in acts.items():
                targets = []
                for r, sp in samples:
                    if sp in Q:
                        max_next = max(Q[sp].values()) if Q[sp] else 0.0
                    else:
                        max_next = 0.0
                    targets.append(r + gamma * max_next)    # loookahead equation
                new_q = np.mean(targets) if targets else 0.0
                old_q = Q[s][a]
                Q_new[s][a] = new_q # update of Q
                delta += abs(new_q - old_q)     
                count += 1
        
        mean_change = delta / max(count,1)
        Q = Q_new

        elapsed = time.time() - start
        print(f"Iter {it:3d} | mean ΔQ={mean_change:.6f} | time={elapsed:.2f}s")
        if mean_change < tol:
            break
    return Q

def q_learning(T,Q, gamma = 0.95, alpha = 1.0, max_iter = 200, tol = 1e-4):
    np.random.seed(42)  # for reproducibility
    # gather all samples
    all_samples = []
    for s, acts in T.items():
        for a, samples in acts.items():
            for r, sp in samples:
                all_samples.append((s, a, r, sp))

    for it in range(1, max_iter+1):
        np.random.shuffle(all_samples)
        delta = 0.0

        for s, a, r, sp in all_samples:
            if sp in Q:
                max_next = max(Q[sp].values())
            else:
                max_next = 0.0
            old_q = Q[s][a]
            Q[s][a] += alpha * (r + gamma * max_next - Q[s][a])   # Q-learning update
            delta += abs(Q[s][a] - old_q)
        
        mean_change = delta / len(all_samples)
        print(f"Iter {it:3d} | mean ΔQ={mean_change:.6f}")
        if mean_change < tol:
            break
    return Q

def build_policy(Q, n_states = 302020, n_actions = 9, default_action = -1):
    # initialize policy vector
    policy = [-1] * (n_states + 1)

    # for each observed state, fill policy with best action
    observed_actions = {}
    for s, acts in Q.items():
        best_a = 0
        best_q = -float("inf")
        for a in range(1,n_actions+1):
            q = acts.get(a, -float("inf"))
            if q > best_q:
                best_q = q
                best_a = a
        if isinstance(s, int) and 1 <= s < n_states+1:
            observed_actions[s] = best_a
    
    # Fill policy for observed states
    for s, a in observed_actions.items():
        policy[s] = a

    policy = policy[1:]  # adjust for 1-based indexing

    # Nearest neighbor filling
    # For every unobserved state, find neaerest observed state by index
    for s in range(n_states):
        if policy[s] == -1 and s+1 not in observed_actions:
            # find insertion point
            pos = bisect.bisect_left(sorted(observed_actions.keys()), s+1)

            # Search outward from the insertion point for the nearest observed state
            left_idx = pos - 1
            right_idx = pos
            chosen = None
            observed_states = sorted(observed_actions.keys())
            while left_idx >= 0 or right_idx < len(observed_states):
                left_state = observed_states[left_idx] if left_idx >= 0 else None
                right_state = observed_states[right_idx] if right_idx < len(observed_states) else None

                # compute distances, prefer smaller distance
                if left_state is not None and right_state is not None:
                    d_left = (s+1) - left_state
                    d_right = right_state - (s+1)
                    if d_left <= d_right:
                        cand_state = left_state
                        left_idx -= 1
                    else:
                        cand_state = right_state
                        right_idx += 1
                elif left_state is not None:
                    cand_state = left_state
                    left_idx -= 1
                elif right_state is not None:
                    cand_state = right_state
                    right_idx += 1
                else:
                    break

                chosen = cand_state
                break

            if chosen is not None:
                policy[s] = observed_actions[chosen]
            else:
                policy[s] = default_action

    return policy

# START
df = pd.read_csv("data/large.csv")
mark = time.time()
T = build_model(df)
print(f"Model built in {time.time() - mark:.2f}s")
mark = time.time()
Q = initialize_Q(T)
Q = fitted_q_iteration(T,Q,gamma=GAMMA, max_iter=MAX_ITER, tol=TOL)
print(f"Fitted Q-iteration completed in {time.time() - mark:.2f}s")
mark = time.time()
policy = build_policy(Q, n_states=N_STATES, n_actions=N_ACTIONS, default_action=DEFAULT)
print(f"Policy built in {time.time() - mark:.2f}s")

with open("large.policy", "w") as f:
    for a in policy:
        f.write(f"{a}\n")