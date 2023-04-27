import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import random
import time

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(4, 170)
        self.output = nn.Linear(170, 2)
        self.probab = nn.Softmax(dim=0)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.output(x)
        x = self.probab(x)
        return x

class Agent:
    def __init__(self, environment, model=Policy, gamma=0.99, lr=0.0009):
        # Environment features
        self.n_actions = environment.action_space.n
        self.action_space = np.array([*range(environment.action_space.n)])
        self.gamma = gamma
        # Trace the training history
        self.train_rewards = []
        # Initialize the policy model
        self.model = model()
        self.model.eval()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

    def decide(self, state):
        with torch.no_grad():
            action_probs = self.model(torch.from_numpy(state).float())
        action = np.random.choice(self.action_space, p=action_probs.data.numpy())
        return action

    def discount_rewards(self, rewards):
        times = torch.arange(len(rewards)).float()
        disc_rewards = torch.pow(self.gamma, times) * rewards
        disc_rewards /= disc_rewards.max() # Normalize to improve numerical stability
        return disc_rewards

    def __learn(self, transitions):
        # Set model in train mode
        self.model.train()
        # Convert lists to arrays
        states = torch.tensor(np.float32(transitions[0]))
        actions = torch.tensor(np.int64(transitions[1]))
        rewards = torch.tensor(np.float32(transitions[2]))
        # Calculate total rewards
        returns = self.discount_rewards(rewards.flip(dims=(0,))).view(-1,1)
        # Recomputes the action-probabilities for all the states in the episode
        probs = self.model(states)
        # Select the predicted probabilities of the actions that were actually taken
        action_probs = probs.gather(dim=1, index=actions.long().view(-1,1)).squeeze()
        # Calculates the loss
        loss = -torch.sum(torch.log(action_probs) * returns)
        # Zero the gradient
        self.optimizer.zero_grad()
        # Backward propagate the loss
        loss.backward()
        # Adjust model parameters
        self.optimizer.step()
        # Restore the evaluation mode for the model
        self.model.eval()

    def train(self, environment, n_episodes):
        print('>>> Start training')
        self.model.train()
        for i in range(n_episodes):
            self.train_rewards.append(0.0) # Initialize the episode reward
            curr_state = environment.reset()[0] # Reset the environment and obtain the initial state
            done = False # Will be True when the episode is end
            transitions = [[], [], []]
            while not done:
                # Take an action
                action = self.decide(curr_state)
                prev_state = deepcopy(curr_state)
                curr_state, reward, term, trunc = environment.step(action)[:4]
                done = term or trunc
                # Trace the performance of the episode
                self.train_rewards[-1] += reward
                # Save the history of the episode
                transitions[0].append(prev_state)
                transitions[1].append(action)
                transitions[2].append(self.train_rewards[-1])
            # Learn from the episode
            self.__learn(transitions)
            if i % 10 == 0:
                print('# Episode:', i)
        self.model.eval()
        print('>>> End training')
    
    def train_history(self, bin_window=0):
        """
        Train history

        :param bin_window: integer, size of the window within which computing the mean.
        :return: numeric list, total reward obtained at each training episode, sorted in time order.
        """
        if bin_window < 2:
            train_rewards = self.train_rewards
        else:
            train_rewards = bin_mean(self.train_rewards, window=bin_window)
        return train_rewards

def bin_mean(x, window=10):
    """
    Split a time series in bins and returns the mean value for each bin.

    :param x: a time series.
    :param window: integer, size of the window within which computing the mean.
    """
    x_size = len(x)
    n_bins = int(np.ceil(x_size/window))
    one_hot = np.eye(n_bins)
    kernel = []
    for i in range(n_bins):
        kernel.append(np.array([]))
        for j in range(n_bins):
            one_hot_expanded = np.repeat(one_hot[i][j], window)
            kernel[i] = np.append(kernel[i], one_hot_expanded)
    kernel = np.stack(kernel, axis=1)[:x_size,:]
    bin_size = np.sum(kernel, axis=0)
    return x @ kernel / bin_size

def play(agent, sleep=0, random_state=None, render=False):
    """
    Play with the agent: walk on lands!

    :param agent: an agent of Autopilot class.
    :param sleep: number of seconds elapsing between frames.
    :param random_state: seed for the environment generation.
    :param render: boolean, specifies if render the episode.
    """
    render_mode = None if render is False else 'human'
    environment = gym.make('CartPole-v1', render_mode=render_mode)
    state = environment.reset(seed=random_state)[0]
    if render == True:
        environment.render()
    else:
        sleep = 0
    done = False
    episode_reward = 0
    duration = 0
    while not done:
        duration += 1
        time.sleep(sleep)
        action = agent.decide(state)
        state, reward, term, trunc = environment.step(action)[:4]
        done = term or trunc or (duration == 500)
        episode_reward += reward
        if render == True:
            environment.render()
    if render == True:
        time.sleep(sleep * 2)
        environment.close()
    return {'duration': duration, 'reward': episode_reward, 'solved': episode_reward >= 500}
