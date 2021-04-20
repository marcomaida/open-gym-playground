import gym
import numpy as np
import numpy.random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

############## NN

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        neurons = 30
        self.network = nn.Sequential(
            nn.Linear(5, neurons), # 4 for observations + 1 for action
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons-10),
            nn.ReLU(),
            nn.Linear(neurons-10, 1),
            nn.ReLU()
        )

    def forward(self, x):   
        result = self.network(x.float())
        return result

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=.0001)

def train(env, model, loss_fn, optimizer, epoch):
    observation = env.reset()
    replay_size = 2000
    size = 2000
    eps = .4
    discount = .05
    replay = []
    for i in range(size+1):
        # Using env to get X and y
        if np.random.rand() > eps:
            action = decide(model, observation)
        else:   
            action = env.action_space.sample() # Random action

        X = torch.tensor([observation[i] for i in range(4)] + [float(action)])
        observation, reward, done, info = env.step(action)

        if epoch == 0:
            discount_now = i / size * discount 
        else:
            discount_now = discount

        # Finding q
        X_next = torch.tensor([observation[i] for i in range(4)] + [0.])
        q = reward + discount_now * model(X_next)[0]
        X_next = torch.tensor([observation[i] for i in range(4)] + [1.])
        q = min(q, reward + discount_now * model(X_next)[0])

        if reward == 0: q = 0

        y = torch.tensor([q])

        transition = (X,y)

        replay.insert(0, transition)
        replay = replay[:replay_size]

        if done:
            observation = env.reset()

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction error
        if len(replay) > 100:
            for _ in range(10):
                X,y = replay[np.random.randint(len(replay))]
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), i
            print(f"loss: {loss:.15f}  [{current:>5d}/{size:>5d}]")

def decide(network, observation):
    X = torch.tensor([observation[i] for i in range(4)] + [0.])
    left_state = network(X)
    X[4] = 1.
    right_state = network(X)

    action = 1 if abs(right_state[0]) > abs(left_state[0]) else 0

    return action

def test(env, model):
    observation = env.reset()
    model.eval()

    episodes = 10
    for i in range(episodes):
        total_reward = 0
        done = False
        while not done:
            # env.render()
            action = decide(model, observation)

            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                observation = env.reset()
                print(f"episode {i}, reward: {total_reward}")

############## Application

env = gym.make("CartPole-v1")
observation = env.reset()

print (env.action_space)
print (env.observation_space)

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(env, model, loss_fn, optimizer, t)
    test(env, model)

print("Done!")
