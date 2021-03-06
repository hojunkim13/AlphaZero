from Config import *

Black = 0
White = 1
player = {0: "Black", 1: "White"}

dir_dict = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
}


def get_neighbor_move(move, d):
    if move is None:
        return None
    r, c = divmod(move, BOARD_SIZE)
    row, col = [i + j for i, j in zip([r, c], dir_dict[d])]
    if min(row, col) < 0 or max(row, col) >= BOARD_SIZE:
        return
    neighbor_move = row * BOARD_SIZE + col
    return neighbor_move


class Gomoku:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_player = Black
        self.board = {}
        self.empty = list(range(BOARD_SIZE ** 2))

    def play(self, move):
        if not self.is_legal(move):
            return
        self.board[move] = self.current_player
        self.empty.remove(move)
        self.current_player ^= 1

    def get_legal_moves(self):
        return self.empty.copy()

    def is_legal(self, move):
        return move in self.empty

    def is_done(self):
        minimun_turn = WIN_CONDITION * 2 - 1
        if len(self.board) < minimun_turn:
            return False, -1

        for m, p in self.board.items():
            for d in range(4):
                ps = set([p])
                m_ = m
                for i in range(WIN_CONDITION - 1):
                    m_ = get_neighbor_move(m_, d)
                    ps.add(self.board.get(m_, -1))
                if len(ps) == 1:
                    return True, p

        if not self.empty:
            return True, -1

        return False, -1

    def __str__(self):
        s = "  "
        for i in range(BOARD_SIZE):
            s += f"{i}".center(2)
        s += "\n"
        for row in range(BOARD_SIZE):
            s += "{}".format(row).center(2)
            for col in range(BOARD_SIZE):
                move = row * BOARD_SIZE + col
                player = self.board.get(move, -1)
                if player == 0:
                    s += "○".center(2)
                elif player == 1:
                    s += "●".center(2)
                else:
                    s += "∙".center(2)  # ◦▫▪
            s += "\n"
        return s


def human_action(state):
    while True:
        try:
            action = input("Move : ").split(":")[-1].split(" ")
            row, col = action
            row, col = int(row), int(col)
            action = row * BOARD_SIZE + col
            break
        except KeyboardInterrupt:
            quit()
        except:
            pass
    return action


if __name__ == "__main__":
    game = Gomoku()
    while True:
        done, winner = game.is_done()
        if done:
            break
        print(game)
        game.play(int(input("Move : ")))
    print(game)
    print(f"Winner : {winner}")
