from copy import deepcopy
import numpy as np

from game import BOARD_SIZE


class Node:
    def __init__(self, parent, state, p):
        self.parent = parent
        self.state = deepcopy(state)
        self.p = p
        self.w = 0
        self.n = 0
        self.childs = []
        self.legal_moves = state.get_legal_moves()

    def get_puct(self, c_puct=1):
        q = -self.w / (self.n + 1e-8)
        u = c_puct * self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return q + u

    def is_leaf(self):
        return not self.childs

    def is_root(self):
        return self.parent is None


class MCTS:
    def __init__(self, net):
        self.net = net

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
            if winner == -1:
                value = 0
            else:
                value = 1 if winner == leaf_node.state.current_player else -1
        else:
            prob, value = self.net.predict(leaf_node.state)
            self.expand(leaf_node, prob)
        self.backpropagate(leaf_node, value)

    def get_move(self, state, temp=1, n_search=50):
        self.root_node = Node(None, state, 1)
        for _ in range(n_search):
            self.search()

        visits = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for m, child in zip(self.root_node.legal_moves, self.root_node.childs):
            visits[m] = child.n
        probs = visits / visits.sum()
        action = np.random.choice(len(probs), p=probs)
        return action, probs

