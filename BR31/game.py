"""
Baskin Robbins 31 Game

This is a simple game in korea.
Starting with number 1, one person can say the numbers
up to three and last person who speaks 31 becomes a loser.

There is a certain victory strategy in this game. so if the first
player knows this strategy, he can always win. (only for 1 vs 1)
"""
import random

Black = 0
White = 1
player = {0: "Black", 1: "White"}


class BR31:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.turn = Black
        self.n = 0

    def play(self, n):
        n = max(min(n, 2), 0)
        self.n += n + 1
        self.turn ^= 1

    def is_done(self):
        return self.n >= 31

    def __str__(self):
        s = f"Player : {player[self.turn]}, State : {self.n}"
        return s


def strong_player(game, level=1.0):
    if random.random() < level:
        remainder = game.n % 4
        if not remainder:
            action = 2
        else:
            action = remainder
    else:
        action = random.randint(1, 3)
    return action - 1


if __name__ == "__main__":
    game = BR31()
    while not game.is_done():
        print(game)
        game.play(random.randint(1, 3))
    print(f"Winner : {player[game.turn]}")
