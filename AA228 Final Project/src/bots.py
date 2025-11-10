import random
from src.players import Player

class RandomBot(Player):
    def act(self, game):
        """Makes a random valid action."""
        # game exposes current bid (q,face) and legal next bids
        prev_bid, prev_face = game.curent_bid
        
        # choose either to call or to make a legal higher bid


        return 0