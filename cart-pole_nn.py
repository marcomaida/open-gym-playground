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
        layers = 30
        layers_end = 20
        self.network = nn.Sequential(
            nn.Linear(5, layers), # 4 for observations + 1 for action
            nn.ReLU(),
            nn.Linear(layers, layers),
            nn.ReLU(),
            nn.Linear(layers, 45),
            nn.ReLU(),
            nn.Linear(45, layers),
            nn.ReLU(),
            nn.Linear(layers, layers),
            nn.ReLU(),
            nn.Linear(layers, layers_end),
            nn.ReLU(),
            nn.Linear(layers_end, 1),
            nn.ReLU()
        )

    def forward(self, x):   
        result = self.network(x.float())
        return result

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=.0001)

def train(env, model, loss_fn, optimizer):
    observation = env.reset()
    size = 10000
    for i in range(size+1):
        # Using env to get X and y
        if np.random.rand() > .5:
            action = decide(model, observation)
        else:   
            action = env.action_space.sample() # Random action

        X = torch.tensor([observation[i] for i in range(4)] + [float(action)])

        observation, reward, done, info = env.step(action)
        y = torch.tensor([reward])
        if done:
            observation = env.reset()

        # Compute prediction error
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
    left_reward = network(X)
    X[4] = 1.
    right_reward = network(X)
    action = 1 if right_reward > left_reward else 0
    return action

def test(env, model):
    observation = env.reset()
    model.eval()

    for i in range(500):
        env.render()

        action = decide(model, observation)

        observation, reward, done, info = env.step(action)
        if done:
           observation = env.reset()

############## Application

env = gym.make("CartPole-v1")
observation = env.reset()

print (env.action_space)
print (env.observation_space)

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(env, model, loss_fn, optimizer)
    test(env, model)

print("Done!")
