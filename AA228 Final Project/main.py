import numpy as np
from typing import Dict, List, Tuple
from src.game import LiarsDiceGame
from src.bots import RandomBot
from src.state import *

## TEST
# create players: human + 3 random bots
players = [None] * 4
players[0] = None # human player in GUI
for i in range(1, 4):
    players[i] = RandomBot(i)
game = LiarsDiceGame(players)
game.deal()
print("Dealt dice:", game.dice)