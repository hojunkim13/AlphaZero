from copy import deepcopy
import numpy as np
from Config import *
from game import BOARD_SIZE
import os
from datetime import datetime


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
    def __init__(self, net, log=False):
        self.net = net.eval()
        self.logger = MCTSLogger(self) if log else None

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
        node = self.root_node
        leaf_node = self.select(node)
        done, winner = leaf_node.state.is_done()
        if done:
            value = 0 if winner == -1 else -1
        else:
            prob, value = self.net.predict(leaf_node.state)
            self.expand(leaf_node, prob)
        self.backpropagate(leaf_node, value)

    def get_move(self, state, temp=TEMPERATURE):
        self.root_node = Node(None, state, 1)
        for _ in range(N_SEARCH):
            self.search()

        visits = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for m, child in zip(self.root_node.legal_moves, self.root_node.childs):
            visits[m] = child.n
        probs = visits / visits.sum()

        if not temp:
            action = np.argmax(probs)
        else:
            action = np.random.choice(len(probs), p=probs)
        if self.logger:
            self.logger.log(action, probs)
        return action, probs


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

