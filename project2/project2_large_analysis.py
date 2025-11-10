import pandas as pd
import numpy as np
import json
import sys
from typing import Dict, Any

# load csv
df = pd.read_csv("data/large.csv")

def rewards_per_action(df: pd.DataFrame, action_col: str = 'a', reward_col: str = 'r') -> Dict[Any, list]:
	if action_col not in df.columns or reward_col not in df.columns:
		raise KeyError(f"DataFrame must contain columns '{action_col}' and '{reward_col}'")

	# group by action and collect unique rewards; convert numpy arrays to Python lists
	grouped = df.groupby(action_col)[reward_col].unique()
	result = {int(k) if (isinstance(k, (np.integer,)) or (isinstance(k, float) and k.is_integer())) else k: 
			  (np.sort(v).tolist() if hasattr(v, 'tolist') else sorted(list(set(v))))
			  for k, v in grouped.items()}

	return result


def transitions_from_state(df: pd.DataFrame, state, state_col: str = 's', action_col: str = 'a', reward_col: str = 'r', next_state_col: str = 'sp') -> Dict[Any, list]:
	for col in (state_col, action_col, reward_col, next_state_col):
		if col not in df.columns:
			raise KeyError(f"DataFrame must contain column '{col}'")

	sub = df[df[state_col] == state]
	if sub.empty:
		return {}

	grouped: Dict[Any, list] = {}
	for act, g in sub.groupby(action_col):
		# drop duplicate (reward, next_state) pairs
		pairs = g[[reward_col, next_state_col]].drop_duplicates()
		# convert to list of simple dicts for JSON serialization
		records = [{"r": (float(row[reward_col]) if np.isscalar(row[reward_col]) else row[reward_col]),
				"sp": (int(row[next_state_col]) if (isinstance(row[next_state_col], (np.integer,)) or (isinstance(row[next_state_col], float) and float(row[next_state_col]).is_integer())) else row[next_state_col])}
			   for _, row in pairs.iterrows()]
		# sort records optionally by next-state then reward to keep deterministic order
		try:
			records.sort(key=lambda x: (x['sp'], x['r']))
		except Exception:
			pass
		grouped[int(act) if (isinstance(act, (np.integer,)) or (isinstance(act, float) and float(act).is_integer())) else act] = records

	return grouped


if __name__ == '__main__':
	try:
		rewards = rewards_per_action(df)
	except KeyError as e:
		print(f"Error: {e}")
		raise

	# Print a concise summary
	print("Rewards per action summary:")
	for a, rs in sorted(rewards.items(), key=lambda x: x[0]):
		sample = rs[:10]
		print(f"Action {a}: {len(rs)} unique rewards, sample: {sample}")

	# --- transitions extraction for a given state ---
	# If a state is provided as a command-line argument, use it; otherwise default to first state
	state_arg = None
	if len(sys.argv) > 1:
		raw = sys.argv[1]
		try:
			if '.' in raw:
				state_arg = float(raw)
				if isinstance(state_arg, float) and state_arg.is_integer():
					state_arg = int(state_arg)
			else:
				state_arg = int(raw)
		except ValueError:
			state_arg = raw
	else:
		state_arg = df['s'].iloc[0] if 's' in df.columns and not df['s'].empty else None

	if state_arg is not None:
		transitions = transitions_from_state(df, state_arg)

		# Print concise summary
		if transitions:
			print(f"Transitions from state {state_arg}: {len(transitions)} actions found")
			for a, lst in sorted(transitions.items(), key=lambda x: x[0]):
				sample = lst[:5]
				print(f" Action {a}: {len(lst)} unique (r,sp) pairs, sample: {sample}")
		else:
			print(f"No transitions found from state {state_arg}")