from copy import deepcopy
import numpy as np
from Config import *
from game import BOARD_SIZE
import os
from datetime import datetime


class MCTSLogger:
    def __init__(self, mcts) -> None:
        self.mcts = mcts
        title = datetime.strftime(datetime.now(), "%m%d-%H_%M")
        os.makedirs("./Gomoku/log/", exist_ok=True)
        self.path = f"./Gomoku/log/{title}.log"

    def log(self, action, prob):
        game = self.mcts.root_node.state

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
        log += f"# Move: {conv_move(action)}\n"
        log += f"# Legal move  : {list(move_prob_dict.keys())[:5]}\n"
        log += f"# Probability : {list(move_prob_dict.values())[:5]}\n"
        with open(self.path, mode="a", encoding="utf-8") as file:
            file.write(log + str(game) + "\n\n")


class Node:
    def __init__(self, parent, state, p):
        self.parent = parent
        self.state = deepcopy(state)
        self.p = p
        self.w = 0
        self.n = 0
        self.childs = []
        self.legal_moves = state.get_legal_moves()

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
        self.net = net.eval()
        self.logger = MCTSLogger(self) if log else None
        self.is_self_play = self_play
        self.last_move = None

    def select(self, node):
        while not node.is_leaf():
            best_puct = -1e5
            best_child = None
            for n in node.childs:
                puct = n.get_puct()
                if puct > best_puct:
                    best_puct = puct
                    best_child = n
            node = best_child
        return node

    def expand(self, node, probs):
        for move in node.legal_moves:
            p = probs[move]
            new_state = deepcopy(node.state)
            new_state.play(move)
            child = Node(node, new_state, p)
            node.childs.append(child)

    def backpropagate(self, node, value):
        node.w += value
        node.n += 1
        if not node.is_root():
            self.backpropagate(node.parent, -value)

    def search(self):
        leaf_node = self.select(self.root_node)
        done, winner = leaf_node.state.is_done()
        if done:
            value = 0 if winner == -1 else -1
        else:
            prob, value = self.net.predict(leaf_node.state)
            self.expand(leaf_node, prob)
        self.backpropagate(leaf_node, value)

    def get_move(self, state, temp=TEMPERATURE):
        if self.is_self_play and self.last_move is not None:
            idx = self.root_node.legal_moves.index(self.last_move)
            self.root_node = self.root_node.childs[idx]
            self.root_node.parent = None
        else:
            self.root_node = Node(None, state, 1)
        for _ in range(N_SEARCH):
            self.search()

        legal_moves = self.root_node.legal_moves
        legal_visits = np.array([c.n for c in self.root_node.childs])

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
            self.logger.log(move, all_probs)
        self.last_move = move
        return move, all_probs

