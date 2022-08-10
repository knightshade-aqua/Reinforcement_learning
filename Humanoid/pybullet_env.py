#python pybullet_env.py

import gym
from pip import main  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np
import time
#import pybullet_envs
import matplotlib.pyplot as plt 
from utils import NormalizedActions, plot
from sac_networks import ValueNetwork, ActorNetwork, CriticNetwork
import torch
import torch.nn as nn
from torch import optim
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from sac_update import update

def main():
    NUM_EPISODES = 100
    MAX_TIMESTEPS = 500
    GLOBAL_COUNT = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-4
    frame_idx = 0 
    max_frames = 10000
    batch_size = 64
    writer = SummaryWriter()
    gamma = 0.99
    soft_tau = 1e-2
    entropy_alpha = 0.2
    rewards     = []
    
    
    env = NormalizedActions(gym.make('HumanoidPyBulletEnv-v0'))
    state_dim = env.observation_space.shape[0] #(44,)
    action_dim = env.action_space.shape[0] #(17,)
    hidden_dim = 256

    value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

    # Double Q network
    critic_net_1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
    critic_net_2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)

    # Policy network
    actor_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_value_net_param, value_net_param in zip(target_value_net.parameters(), value_net.parameters()):
        target_value_net_param.data.copy_(value_net_param.data)

    # Define losses
    value_criterion  = nn.MSELoss()
    critic_net_1_criterion = nn.MSELoss()
    critic_net_2_criterion = nn.MSELoss()

    # Define Optimizer
    value_optimizer  = optim.Adam(value_net.parameters(), lr=lr)
    critic_net_1_optimizer = optim.Adam(critic_net_1.parameters(), lr=lr)
    critic_net_2_optimizer = optim.Adam(critic_net_1.parameters(), lr=lr)
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=lr)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    env.render(mode = "human")


    while frame_idx < max_frames:
        
        state = env.reset()
        episode_reward = 0
        
        for t in range(MAX_TIMESTEPS):

            if frame_idx >1000:
                action = actor_net.get_action(state).detach()
                next_state, reward, done, _ = env.step(action.numpy())
                #print(f"The reward is : {reward}")
            else:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            frame_idx += 1
            print(f"The frame index is : {frame_idx}")

            if len(replay_buffer) > batch_size:
                update(batch_size, device, replay_buffer, \
                        value_net, actor_net, critic_net_1, critic_net_2, target_value_net,\
                        value_optimizer, actor_optimizer, critic_net_1_optimizer, critic_net_2_optimizer, \
                        value_criterion, critic_net_1_criterion, critic_net_2_criterion, writer, frame_idx, entropy_alpha, gamma, soft_tau)

            if done:
                break

        print(f"The episode reward is : {episode_reward}")
        writer.add_scalar("Reward", episode_reward, frame_idx)
        rewards.append(episode_reward)
    plot(rewards)

if __name__ == "__main__":
    main()