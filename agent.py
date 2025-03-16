import random
from collections import deque

import torch
import numpy as np

from game import SnakeGameAI, Direction, Point
from utils.constants import *
from model import Linear_QNet, QTrainer
from utils.helper import plot

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = DISCOUNT_RATE # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft on memory full

        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        tail = game.snake[-1]

        # Check Blocks around the head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Bool values to check current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        # List of States
        state = [

            # Danger: Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger: Right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger: Left
            (dir_r and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)),


            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Direction
            game.food.x < game.head.x, # Food Left
            game.food.x > game.head.x, # Food Right
            game.food.y > game.head.y, # Food Up
            game.food.y < game.head.y, # Food Down

            # Distance Metrics to food
            abs(game.food.x - head.x) / game.w, # normalize
            abs(game.food.y - head.y) / game.h,  # normalize

            # Tail Information
            abs(tail.x - head.x) / game.w, # X distance of head to tail
            abs(tail.y - head.y) / game.h, # Y distance of head to tail

            # Path to tail checks
            not game.is_path_blocked(head, tail),
            
            # Distance to the walls
            head.x / game.w, # Dist to left wall
            (game.w - head.x) / game.w, # Dist to right wall
            head.y / game.h, # Dist to top wall
            (game.h - head.y) / game.h, # Dist to bottom wall

            game.detect_loop_pattern(game.recent_moves)
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):

        # Append to memeory to deque
        self.memory.append((state, action, reward, next_state, done)) # Append as single tuple

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Returns tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random Moves: tradeoff exploration / exploitation
        self.epsilon = INITIAL_EXPLORATION_VALUE  - self.n_games

        # Bounded Decay Function
        # self.epsilon = max(MIN_EPSILON, INITIAL_EPSILON - DECAY_RATE * self.n_games)

        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            
            prediction = self.model(state0) # gives raw float values, bound to max and 0-1
            
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # Get Move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (train again on previously played game)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = max(record, score) # update best score
                # TODO: Save model
                agent.model.save()
            
            print(f"Game: {agent.n_games} Score: {score} Record: {record}")
            
            # Append score and mean score for plotting
            plot_score.append(score)

            total_score += score
            mean_score = total_score / agent.n_games

            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score, record)


if __name__ == '__main__':
    train()