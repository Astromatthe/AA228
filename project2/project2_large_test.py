"""
batch_offline_rl.py

Reads a CSV file named 'transitions.csv' with columns: s,a,r,sp
- s and sp are integers like 290112 meaning three two-digit components: AB C D E F -> [AB,CD,EF]
- a is action id (1..9)
- r is reward (numeric)

Outputs: 'optimal_actions.txt' with lines "s,optimal_action" for every state in the full Cartesian product
(using domain lists defined below).

Only uses pandas and numpy.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import sys

# ---------------------------
# USER / PROBLEM SETTINGS
# ---------------------------
CSV_FILE = "data/large.csv"        # input csv (s,a,r,sp)
OUTPUT_FILE = "large.txt" # output mapping s -> best action

# Discount factor
GAMMA = 0.99

# Value-iteration tolerance and max iterations
TOL = 1e-6
MAX_ITERS = 5000

# ---------- Domain assumptions ----------
# IMPORTANT:
# If you intended different domain cardinalities than below, change these lists.
# I interpreted "302020" as three domain sizes (30, 20, 20).
# Each component value is represented as a two-digit string (zero-padded).
#
# Example: first component values "01".."30", second "01".."20", third "01".."20".
#
# If you already know exact lists (e.g. first component only {15,23,27,29,30}),
# replace the lists below with those exact two-digit strings.
FIRST_VALUES  = [f"{i:02d}" for i in range(1, 31)]   # "01".."30"  (30 values)
SECOND_VALUES = [f"{i:02d}" for i in range(1, 21)]   # "01".."20"  (20 values)
THIRD_VALUES  = [f"{i:02d}" for i in range(1, 21)]   # "01".."20"  (20 values)
# total states = len(FIRST_VALUES) * len(SECOND_VALUES) * len(THIRD_VALUES)

ACTIONS = list(range(1, 10))  # actions 1..9

# The domain rules you specified (encoded here as helpers)
REWARD_ACTIONS_1_4 = set([-10, -5, 0, 5, 10, 50, 100])  # possible rewards for actions 1-4

# ---------------------------
# helper functions
# ---------------------------

def int_state_to_components(s_int):
    """Convert integer state like 290112 -> tuple of two-digit strings ('29','01','12')."""
    s_str = str(int(s_int)).zfill(6)
    return s_str[0:2], s_str[2:4], s_str[4:6]

def components_to_state(comp):
    """Turn tuple of 3 two-digit strings into int-like string 'AABBCC'."""
    return "".join(comp)

def canonical_state_str(comp):
    """Return canonical state representation used in output (no leading zeros removal)."""
    return components_to_state(comp)

# ---------------------------
# Load data and build empirical model
# ---------------------------

df = pd.read_csv(CSV_FILE, dtype={'s':str, 'a':int, 'r':float, 'sp':str})
# Ensure s and sp are zero-padded to 6 digits (three two-digit parts)
df['s'] = df['s'].apply(lambda x: str(int(x)).zfill(6))
df['sp'] = df['sp'].apply(lambda x: str(int(x)).zfill(6))

# Parse components
df[['s1','s2','s3']] = df['s'].apply(lambda s: pd.Series([s[0:2], s[2:4], s[4:6]]))
df[['sp1','sp2','sp3']] = df['sp'].apply(lambda s: pd.Series([s[0:2], s[2:4], s[4:6]]))

# Stats: number of observed distinct components
observed_first  = sorted(df['s1'].unique())
observed_second = sorted(df['s2'].unique())
observed_third  = sorted(df['s3'].unique())

print(f"Observed first-component values (count {len(observed_first)}): {observed_first[:10]}...")
print(f"Observed second-component values (count {len(observed_second)}): {observed_second[:10]}...")
print(f"Observed third-component values (count {len(observed_third)}): {observed_third[:10]}...")

# Build mapping indexes for all states in the full Cartesian product we will use
FIRST = FIRST_VALUES
SECOND = SECOND_VALUES
THIRD = THIRD_VALUES

n1, n2, n3 = len(FIRST), len(SECOND), len(THIRD)
N_states = n1 * n2 * n3
print(f"Using domain sizes n1={n1}, n2={n2}, n3={n3} -> total states {N_states}")

# Create index maps for components -> integer index for array indexing
idx_first  = {v:i for i,v in enumerate(FIRST)}
idx_second = {v:i for i,v in enumerate(SECOND)}
idx_third  = {v:i for i,v in enumerate(THIRD)}

def comp_to_idx(c1,c2,c3):
    return idx_first[c1], idx_second[c2], idx_third[c3]

# Flattened state index
def flatten_idx(i,j,k):
    return (i * n2 + j) * n3 + k

def unflatten_idx(flat):
    k = flat % n3
    tmp = flat // n3
    j = tmp % n2
    i = tmp // n2
    return i,j,k

# ---------------------------
# Empirical estimates from dataset
# ---------------------------
# We'll estimate:
#   R_hat[s,a] = average reward observed when (s,a) occurs
#   P_hat[(s,a)] = dictionary of observed next-state counts -> normalized to probabilities
#
# For unobserved (s,a) we apply the domain rules you gave as fallback:
# - actions 1-4: lead to neighbor states where first component identical (we emulate by
#   assuming self-loop with small probability and distributed to observed neighbors if any).
# - actions 5-9: lead to self-loop with zero reward unless special patterns (0101 or 2020) observed,
#   for which we produce a uniform random transition over ALL states (approximation).
#
# NOTE: This fallback is intentionally conservative: it prefers self-loop/no reward when data missing.
# ---------------------------

# containers: counts and sums
reward_sum = defaultdict(float)
count_sa = defaultdict(int)
next_counts = defaultdict(lambda: defaultdict(int))

for _, row in df.iterrows():
    s1,s2,s3 = row['s1'], row['s2'], row['s3']
    sp1,sp2,sp3 = row['sp1'], row['sp2'], row['sp3']
    a = int(row['a'])
    r = float(row['r'])

    key = (s1,s2,s3,a)
    reward_sum[key] += r
    count_sa[key] += 1
    next_counts[key][(sp1,sp2,sp3)] += 1

# Compute averages and normalized transition probabilities
R_hat = {}   # key -> average reward
P_hat = {}   # key -> dict(next_comp -> prob)

for key, cnt in count_sa.items():
    R_hat[key] = reward_sum[key] / cnt
    nxt = next_counts[key]
    total = sum(nxt.values())
    P_hat[key] = {ns: v/total for ns,v in nxt.items()}

# ---------------------------
# Helper: getR and getP that apply fallbacks when (s,a) not observed
# ---------------------------
all_states_list = [(a,b,c) for a in FIRST for b in SECOND for c in THIRD]
all_states_flat = {components_to_state(s): flatten_idx(*comp_to_idx(*s)) for s in all_states_list}

def get_R_and_P(s_comp, a):
    """Return (R, Pdict) for a given state components and action a.
    R: scalar (expected reward)
    Pdict: dict mapping next-state-components -> probability
    """
    key = (s_comp[0], s_comp[1], s_comp[2], a)
    if key in R_hat:
        return R_hat[key], P_hat[key]

    # fallback heuristics based on rules you gave
    if 1 <= a <= 4:
        # actions 1-4: lead to neighbor states where the first two-digit number is identical.
        # We approximate: if we have any observed transitions from any state with same first component and action a,
        # use their empirical distribution averaged; otherwise, self-loop with zero reward.
        similar_keys = [k for k in P_hat.keys() if k[0] == s_comp[0] and k[3] == a]
        if similar_keys:
            # average distributions
            accum = defaultdict(float)
            for sk in similar_keys:
                pd = P_hat[sk]
                for ns, p in pd.items():
                    accum[ns] += p
            # normalize
            total = sum(accum.values())
            for ns in list(accum.keys()):
                accum[ns] /= total
            # average reward among those keys (if available)
            rsum = 0.0
            for sk in similar_keys:
                rsum += R_hat.get(sk, 0.0)
            ravg = rsum / len(similar_keys)
            return ravg, dict(accum)
        else:
            # no data: assume self-loop and zero reward
            return 0.0, {s_comp: 1.0}

    else:
        # actions 5-9: lead to no reward and staying in the same state when the last four digits are not 0101 or 2020.
        # Otherwise, they lead to a random state.
        last_four = s_comp[1] + s_comp[2]  # e.g. '0101' or '2020'
        if last_four not in ('0101', '2020'):
            return 0.0, {s_comp: 1.0}
        else:
            # random transition to any state (approximate with uniform over our domain)
            prob = 1.0 / N_states
            uniform = {s: prob for s in all_states_list}
            return 0.0, uniform

# ---------------------------
# Value iteration on empirical MDP
# ---------------------------
# We'll perform value iteration on the full state set (size N_states).
# For memory efficiency we store V as 1D flat numpy array length N_states.
# For each state and action we compute expected Q(s,a) = R(s,a) + gamma * sum_{sp} P(sp|s,a) * V[sp]
# We iterate until sup-norm change < TOL or reach MAX_ITERS.
# ---------------------------

V = np.zeros(N_states, dtype=float)
policy = np.zeros(N_states, dtype=int)

print("Starting value iteration...")

for it in range(MAX_ITERS):
    delta = 0.0
    V_new = np.empty_like(V)
    # iterate over all states
    for i1, c1 in enumerate(FIRST):
        for i2, c2 in enumerate(SECOND):
            for i3, c3 in enumerate(THIRD):
                flat = flatten_idx(i1, i2, i3)
                s_comp = (c1, c2, c3)
                best_q = -np.inf
                best_a = 1
                # evaluate each action
                for a in ACTIONS:
                    r, Pdict = get_R_and_P(s_comp, a)
                    # estimate expected value of next state
                    exp_v = 0.0
                    # Pdict keys are component tuples
                    for ns_comp, p in Pdict.items():
                        # if ns_comp outside domain (rare), treat as self-loop
                        if ns_comp[0] in idx_first and ns_comp[1] in idx_second and ns_comp[2] in idx_third:
                            ni, nj, nk = comp_to_idx(*ns_comp)
                            nf = flatten_idx(ni, nj, nk)
                            exp_v += p * V[nf]
                        else:
                            # unseen comp -> approximate by current state's value
                            exp_v += p * V[flat]
                    q = r + GAMMA * exp_v
                    if q > best_q:
                        best_q = q
                        best_a = a
                V_new[flat] = best_q
                policy[flat] = best_a
                delta = max(delta, abs(V_new[flat] - V[flat]))
    V = V_new
    if (it+1) % 10 == 0:
        print(f" iter {it+1}, delta={delta:.6g}")
    if delta < TOL:
        print(f"Value iteration converged at iteration {it+1} with delta={delta:.6g}")
        break
else:
    print(f"Value iteration reached max iterations {MAX_ITERS}, last delta={delta:.6g}")

# ---------------------------
# Output the policy mapping for every state (in canonical 6-digit form)
# ---------------------------
with open(OUTPUT_FILE, 'w') as f:
    # write header optional
    # f.write("state,action\n")
    for i1, c1 in enumerate(FIRST):
        for i2, c2 in enumerate(SECOND):
            for i3, c3 in enumerate(THIRD):
                flat = flatten_idx(i1, i2, i3)
                s_str = canonical_state_str((c1,c2,c3))
                a = int(policy[flat])
                f.write(f"{s_str},{a}\n")

print(f"Optimal policy written to '{OUTPUT_FILE}' ({N_states} lines).")
