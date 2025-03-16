# Agent Constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
INITIAL_EXPLORATION_VALUE  = 80
DECAY_RATE = 0.1
MIN_EPSILON = 15
DISCOUNT_RATE = 0.99

# Model Constants
INPUT_SIZE = 21 # No. of world inputs
OUTPUT_SIZE = 3 # No. of actions that can be taken i.e Straight, Left, Right
HIDDEN_SIZE = 256 # Hidenn layer size

# Game Constants
BLOCK_SIZE = 20
SPEED = 500
MAX_IMPROVE_TIME = 100
PREVIOUS_MOVES_BUFFER_SIZE = 30