# Game
BOARD_SIZE = 6
WIN_CONDITION = 4

# Main
ITERATION = 500  #

# MCTS
N_SEARCH = 200  #
TEMPERATURE = 1.0
C_PUCT = 2.0

# Train
TRAIN_EPOCH = 100  #
BATCHSIZE = 512
LR = 1e-3

# Play
N_SELF_PLAY = 10  #
EVAL_N_PLAY = 10

# Path
data_path = "./Gomoku/data/"
weight_path = "./Gomoku/weight/"
