import pandas as pd
import numpy as np
import time

# Parameters
GAMMA = 0.99
MAX_ITER = 500
TOL = 1e-4
N_ACTIONS = 7
N_POS, N_VEL = 500, 100
N_STATES = N_POS * N_VEL

def build_model(df):
    # Build transition and reward matrices from data frame

    # Rewards
    grouped_r = df.groupby(['s','a'])['r'].mean().reset_index() # group df by state and action
    R = {(int(row.s), int(row.a)): float(row.r) for row in grouped_r.itertuples(index = False)} # create dictionary

    # Transitions
    counts = df.groupby(['s','a','sp']).size().rename('count').reset_index()  # Count combination of s->a->sp 
    total_counts = counts.groupby(['s', 'a'])['count'].transform('sum') #get 
    counts['prob'] = counts['count'] / total_counts
    
    # build dictionary
    T = {}
    for row in counts.itertuples(index = False):
        key = (int(row.s), int(row.a))  # exctract key of dictionary
        if key not in T:
            T[key] = []
        T[key].append((int(row.sp), float(row.prob)))   # build dictionary

    return R, T

def value_iteration(R, T, gamma = GAMMA, max_iter = MAX_ITER, tol = TOL):
    # performs value_iteration on expected reward dictionary R and transition model dictionary T
    # perform only for observed states
    observed_states = sorted(set(s for (s,_) in R.keys()))
    U = {s: 0.0 for s in observed_states}

    # run value iteration
    for it in range(max_iter):
        delta = 0.0
        newU = {}

        # update value for each state
        for s in observed_states:
            q_values = []
            # Evaluate all actions (uincluding ones not observed)
            for a in range(N_ACTIONS):
                sa = (s,a)
                r = R.get(sa, 0.0)
                if sa in T: 
                    # if state can be reached, compute expected utility of next state
                    exp_u = sum(prob * U.get(sp, 0.0) for sp,prob in T[s,a])
                else:
                    exp_u = 0.0
                q = r + gamma * exp_u   # lookahead equation
                q_values.append(q)

            # greedy: pick best expected value
            newU[s] = max(q_values)
            delta = max(delta, abs(newU[s] - U[s]))

        # Update
        U = newU

        # break if < tol
        if delta < tol:
            break

    return U

def extract_policy(U, R, T):
    # extracts the policy given dictionaries of values U, rewards R, and transition probabilities T
    policy = {}
    observed_states = sorted(set(s for (s,_) in R.keys()))

    for s in observed_states:
        best_a, best_q = None, -np.Inf

        # step through all actions and take greedy action
        for a in range(N_ACTIONS):
            sa = (s,a)
            r = R.get(sa, 0.0)
            if sa in T:
                exp_u = sum(prob * U.get(sp,0.0) for sp, prob in T[sa])
            else:
                exp_u = 0.0
            q = r + GAMMA * exp_u   # lookahead equation
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a + 1  # shift actions to 1-7

    return policy

def save_policy(policy, filename):
    # saves policy dictionary to file
    with open(filename, 'w') as f:
        for s in range(N_STATES):
            a = policy.get(s, 0)  
            if a == 0:
                a = 4  # default action 4 if state not observed
            f.write(f"{a}\n")

# START 
df = pd.read_csv("data/medium.csv")
mark = time.time()
R,T = build_model(df)
print(f"Model built in {time.time() - mark:.2f}s")
mark = time.time()
U = value_iteration(R,T)
print(U)
print(f"Value iteration completed in {time.time() - mark:.2f}s")
mark = time.time()
policy = extract_policy(U,R,T)
print(f"Policy extracted in {time.time() - mark:.2f}s")
save_policy(policy, "medium.policy")