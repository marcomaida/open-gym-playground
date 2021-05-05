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
            nn.Linear(4, 512), # 4 for observations + 1 for action
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
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
    eps = .6
    discount = .01
    replay = []
    for i in range(size+1):
        # Using env to get X and y
        if np.random.rand() > eps:
            action = decide(model, observation)
        else:   
            action = env.action_space.sample() # Random action

        X = torch.tensor([observation[i] for i in range(4)])
        y_now = model(X)
        observation, reward, done, info = env.step(action)

        if epoch == 0:
            discount_now = i / size * discount 
        else:
            discount_now = discount

        # Finding q
        X_next = torch.tensor([observation[i] for i in range(4)])
        q = reward + discount_now * np.min(model(X_next).detach().numpy())
        if reward == 0: q = 0

        if action == 0:
            y = torch.tensor([q, y_now[1]])
        else:
            y = torch.tensor([y_now[0], q])

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
            for _ in range(3):
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
    X = torch.tensor([observation[i] for i in range(4)])
    return np.argmax(model(X).detach().numpy())

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
