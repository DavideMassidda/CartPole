import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(4, 180)
        #self.dense2 = nn.Linear(128, 128)
        self.output = nn.Linear(180, 2)
        self.probab = nn.Softmax(dim=0)

    def forward(self, x):
        #x_orig = x
        x = F.relu(self.dense1(x))
        #x = F.relu(self.dense2(x))
        x = self.output(x)
        x = self.probab(x)
        #if torch.isnan(x).sum() > 0:
        #    pdb.set_trace()
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

    def __get_action(self, act_prob):
        act_prob = act_prob.data.numpy()
        # Replace NaN, is any
        for j in range(self.n_actions):
            if np.isnan(act_prob[j]):
                print('Warning: NaN found')
                act_prob = np.full(self.n_actions, fill_value=0.5/self.n_actions)
                act_prob /= np.sum(act_prob)
                break
        # Choice sampling with the policy
        action = np.random.choice(self.action_space, p=act_prob)
        return action

    def decide(self, state):
        with torch.no_grad():
            act_prob = self.model(torch.from_numpy(state).float())
        return self.__get_action(act_prob)

    def __decide(self, state):
        self.model.train()
        act_prob = self.model(torch.from_numpy(state).float())
        self.model.eval()
        return self.__get_action(act_prob)

    def get_returns_old(self, rewards):
        nt = len(rewards)
        gamma_exp = self.gamma ** np.flip(np.arange(nt))
        disc_rewards = np.zeros(nt)
        for t in range(nt):
            disc_rewards[t] = np.sum(rewards[t:] * gamma_exp[t:])
        disc_rewards /= disc_rewards.max()
        return disc_rewards

    def get_returns(self, rewards):
        times = torch.arange(len(rewards)).float()
        disc_rewards = torch.pow(self.gamma, times) * rewards
        #disc_rewards -= disc_rewards.mean(axis=0)
        disc_rewards /= disc_rewards.max()
        return disc_rewards

    def get_loss(self, preds, disc_rewards):
        return -1 * torch.sum(disc_rewards * torch.log(preds)) # e non torch.sum

    def __learn(self, transitions):
        # Set model in train mode
        self.model.train()
        # Convert lists to arrays
        state_batch = torch.tensor(np.float32(transitions[0])) # state
        action_batch = torch.tensor(np.int64(transitions[1])) # action
        reward_batch = torch.tensor(np.float32(transitions[2])) # reward
        # Calculate total rewards
        return_batch = self.get_returns(reward_batch.flip(dims=(0,)))
        # Recomputes the action-probabilities for all the states in the episode
        pred_batch = self.model(state_batch)
        # Select the predicted probabilities of the actions that were actually taken
        prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()
        # Adjust model weights
        loss = self.get_loss(prob_batch, return_batch)
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
            curr_state = environment.reset() # Reset the environment and obtain the initial state
            done = False # Will be True when the episode is end
            transitions = [[], [], []]
            while not done:
                # Take an action
                action = self.decide(curr_state)
                prev_state = curr_state
                curr_state, reward, done, info = environment.step(action)
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

def play(environment, agent, render=False, sleep=0):
    """
    Play with the agent: land on the Moon!

    :param environment: the LunarLander-v2 gym environment.
    :param agent: an agent of Autopilot class.
    :param render: boolean, specifies if render the episode.
    :param sleep: number of seconds elapsing between frames.
    """
    state = environment.reset()
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
        state, reward, done, _ = environment.step(action)
        episode_reward += reward
        if render == True:
            environment.render()
    if render == True:
        time.sleep(sleep * 2)
        environment.close()
    return {'duration': duration, 'reward': episode_reward, 'solved': episode_reward >= 200}

def test(environment, agent, n_episodes):
    """
    Test the agent performance

    :param environment: the LunarLander-v2 gym environment.
    :param agent: an agent of Autopilot class.
    :param n_episodes: integer, number of episodes to run.
    :return: numeric list, total reward obtained at each training episode, sorted in time order.
    """
    total_rewards = [play(environment, agent) for i in range(n_episodes)]
    return total_rewards
