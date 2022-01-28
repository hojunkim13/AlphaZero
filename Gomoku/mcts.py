import numpy as np
from Config import *
import os
from datetime import datetime
from copy import deepcopy


class MCTSLogger:
    def __init__(self, mcts) -> None:
        self.mcts = mcts
        title = datetime.strftime(datetime.now(), "%m%d-%H_%M")
        os.makedirs("./Gomoku/log/", exist_ok=True)
        self.path = f"./Gomoku/log/{title}.log"

    def log(self, game, move, prob):
        win_rate = (self.mcts.root_node.w / self.mcts.root_node.n + 1) / 2
        legal_move = game.get_legal_moves()
        legal_prob = prob[legal_move]
        conv_move = lambda x: divmod(x, BOARD_SIZE)
        move_prob_dict = {
            conv_move(m): round(p, 2) for m, p in zip(legal_move, legal_prob)
        }
        move_prob_dict = dict(
            sorted(move_prob_dict.items(), key=lambda x: x[1], reverse=True)
        )

        log = f"# Player: {'White' if game.current_player else 'Black'}\t"
        log += f"# Win Rate: {win_rate*1e2:.1f}%\t"
        log += f"# Move: {conv_move(move)}\n"
        log += f"# Legal move  : {list(move_prob_dict.keys())[:5]}\n"
        log += f"# Probability : {list(move_prob_dict.values())[:5]}\n"
        with open(self.path, mode="a", encoding="utf-8") as file:
            file.write(log + str(game) + "\n\n")


class Node:
    def __init__(self, parent, move, p):
        self.parent = parent
        self.move = move
        self.p = p
        self.w = 0
        self.n = 0
        self.childs = {}

    def get_puct(self):
        q = -self.w / (self.n + 1e-8)
        u = C_PUCT * self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return q + u

    def is_leaf(self):
        return not self.childs

    def is_root(self):
        return self.parent is None


class MCTS:
    def __init__(self, net, log=False, self_play=False):
        self.net = net.eval().cpu()
        self.logger = MCTSLogger(self) if log else None
        self.is_self_play = self_play
        self.last_move = None

    def select(self, node):
        best_puct = -1e5
        best_child = None
        for n in node.childs.values():
            puct = n.get_puct()
            if puct > best_puct:
                best_puct = puct
                best_child = n
        return best_child

    def expand(self, node, state, probs):
        for move in state.get_legal_moves():
            p = probs[move]
            child = Node(node, move, p)
            node.childs[move] = child

    def backpropagate(self, node, value):
        node.n += 1
        node.w += value
        if not node.is_root():
            self.backpropagate(node.parent, -value)

    def search(self, state):
        node = self.root_node
        while True:
            if node.is_leaf():
                break
            node = self.select(node)
            state.play(node.move)

        done, winner = state.is_done()
        if done:
            value = 0 if winner == -1 else -1
        else:
            prob, value = self.net.predict(state)
            self.expand(node, state, prob)
        self.backpropagate(node, value)

    def get_move(self, state, temp=TEMPERATURE):
        cond = [not self.is_self_play, not state.board]
        if any(cond):
            self.root_node = Node(None, None, None)

        for _ in range(N_SEARCH):
            self.search(deepcopy(state))

        legal_moves = state.get_legal_moves()
        legal_visits = np.array([c.n for c in self.root_node.childs.values()])

        if not temp:
            legal_probs = legal_visits / legal_visits.sum()
        else:
            legal_probs = (legal_visits / legal_visits.sum()) ** (1 / temp)

        if self.is_self_play:
            move = np.random.choice(
                legal_moves,
                p=0.75 * legal_probs
                + 0.25 * np.random.dirichlet(0.3 * np.ones_like(legal_probs)),
            )
        else:
            if not temp:
                idx = np.argmax(legal_probs)
                move = legal_moves[idx]
            else:
                move = np.random.choice(legal_moves, p=legal_probs)

        all_probs = np.zeros(BOARD_SIZE ** 2)
        for m in range(BOARD_SIZE ** 2):
            try:
                idx = legal_moves.index(m)
                all_probs[m] = legal_probs[idx]
            except ValueError:
                all_probs[m] = 0

        if self.logger:
            self.logger.log(state, move, all_probs)

        self.root_node = self.root_node.childs[move]

        return move, all_probs

