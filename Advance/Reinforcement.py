import warnings
warnings.filterwarnings('ignore')

# Override the default warning function
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.setrecursionlimit(1500)

import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import gymnasium as gym

# Create the environment
env = gym.make('CartPole-v1')

# Set random seed for reproducibility
np.random.seed(42)
env.action_space.seed(42)
env.observation_space.seed(42)

# Define the model building function
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Input(shape=(state_size,)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# creating the environment and set up the model
env = gym.make('CartPole-v1')
state_size = int(env.observation_space.shape[0])
action_size = int(env.action_space.n)
model = build_model(state_size, action_size)


# Implementing the Q-Learning algorithm

# Define the replay function
# Define epsilon and epsilon_decay
epsilon = 1.0
epsilon_min = 0.01
decay_rate = 0.99

# Replay memory
memory = deque(maxlen=2000)

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def replay(batch_size=64):
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)

    # Extract information for batch processing
    states = np.vstack([x[0] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_states = np.vstack([x[3] for x in minibatch])
    dones = np.array([x[4] for x in minibatch])

    # Predict Q-values for the next states in batch
    q_next = model.predict(next_states)
    # Predict the Q-values for the current states in batch
    q_target = model.predict(states)

    # Vectorized update of target values
    for i in range(batch_size):
        target = rewards[i]
        if not dones[i]:
            # Update Q value with the discounted future reward
            target += 0.95 * np.max(q_next[i])
            # Update only the taken action's Q value
            q_target[i][actions[i]] = target

    # Train the model with the updated targets in batch
    model.fit(states, q_target, epochs=1, verbose=0)

    # Reduce exploration rate (epsilon) after each step
    global epsilon
    if epsilon > epsilon_min:
        epsilon *= decay_rate

def act(state):
    """Choose an action based on the current state and exploration rate."""

    if np.random.rand() < epsilon:
        # Explore: choose a random action
        return np.random.randint(action_size)
    # Exploit: predict action based on the state
    act_values = model.predict(state)
    # Return the action with the highest Q-value
    return np.argmax(act_values[0])

# Define the number of episodes you want to train the model for
episodes = 200
train_freq = 5

for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(200):
        action = act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print(f"episode: {e+1}/{episodes} score: {time}, e: {epsilon:.2}")
            break

        # Train the model every 'train_frequency' steps
        if time % train_freq == 0:
            # Call replay with larger batch size for efficiency
            replay(batch_size=64)

env.close()

#Evalute the Performance
for e in range(10):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()
        action = np.argmax(model.predict(state)[0])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            print(f"episode: {e+1}/10, score: {time}")
            break
env.close()