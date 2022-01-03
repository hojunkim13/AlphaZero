from copy import deepcopy

Black = 0
White = 1
player = {0: "Black", 1: "White"}

BOARD_SIZE = 3
WIN_CONDITION = 3
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


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_player = Black
        self.board = {}
        self.empty = list(range(BOARD_SIZE ** 2))

    def play(self, move):
        done, _ = self.is_done()
        if done:
            return

        if self.is_legal(move):
            self.board[move] = self.current_player
            self.empty.remove(move)
            self.current_player ^= 1

    def get_legal_moves(self):
        return self.empty

    def is_legal(self, move):
        return move in self.empty

    def is_done(self):
        if not self.board:
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
            action = input("Move : ").replace(" ", "").split(":")[-1]
            row, col = action
            row, col = int(row), int(col)
            action = row * BOARD_SIZE + col
            break
        except:
            pass
    return action


def alpha_beta(game, alpha, beta):
    done, winner = game.is_done()
    if done:
        if winner == -1:
            return 0
        else:
            return -1

    for action in game.get_legal_moves():
        game_ = deepcopy(game)
        game_.play(action)
        score = -alpha_beta(game_, -beta, -alpha)
        if score > alpha:
            alpha = score

        if alpha >= beta:
            return alpha

    return alpha


def alpha_beta_player(game):
    game = deepcopy(game)
    best_action = 0
    alpha = -1e3
    for action in game.get_legal_moves():
        game_ = deepcopy(game)
        game_.play(action)
        score = -alpha_beta(game_, -1e3, -alpha)
        if score > alpha:
            best_action = action
            alpha = score

    return best_action


if __name__ == "__main__":
    game = TicTacToe()
    while True:
        done, winner = game.is_done()
        if done:
            break
        print(game)
        game.play(int(input("Move : ")))
    print(game)
    print(f"Winner : {winner}")
