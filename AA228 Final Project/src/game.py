from config import *
from src.rules import *
import numpy as np

class LiarsDiceGame:

    def __init__(self, players):
        self.players = players
        self.total_dice = TOTAL_DICE
        self.current_bid = [0, 0]  # quantity, face
        self.history = []  # to store history of bids and actions
    
    def deal(self):
        """Deal dice to players."""
        # full deal: each player gets DICE_PER_PLAYER dice
        self.dice = [list(np.random.randint(1, FACE_COUNT + 1, DICE_PER_PLAYER)) for _ in range(N_PLAYERS)]

    def step(self, actor_id, action):
        # action is ("call", None) or ("bid", (q, f))
        if action[0] == "bid":
            q, f = action[1]
            self.current_bid = [q, f]
            self.history.append((actor_id, ("bid", (q, f))))
            return None
        else:
            # resolve call
            q, f = self.current_bid
            actual = count_face_total(self.dice, f)
            caller = actor_id
            last_bidder = (self.history[-1][0]) if self.history else None
            result = {
                "bid": (q, f),
                "actual": actual,
                "winner": None
            }
            if actual >= q:
                # last bidder wins
                result["winner"] = last_bidder
            else:
                # caller wins
                result["winner"] = caller
            self.history.append((actor_id, ("call", None)))
            return result