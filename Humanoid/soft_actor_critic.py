#python soft_actor_critic.py

import gym
import torch.nn as nn
import random
from torch import optim
import numpy as np
import numpy
from sac_networks import ValueNetwork, SoftQNetwork, PolicyNetwork
from replay_buffer import ReplayBuffer
from normal_actions import NormalizedActions
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal
import math
from matplotlib import animation
from torch.utils.tensorboard import SummaryWriter

#from NormalizedActions import NormalizedActions


def plot(rewards):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    #plt.subplot(131)
    #plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def update(batch_size, device, replay_buffer, soft_q_net1, soft_q_net2, \
            value_net, policy_net, target_value_net, soft_q_criterion1, soft_q_criterion2, soft_q_optimizer1, soft_q_optimizer2, value_optimizer, \
            policy_optimizer, value_criterion, writer, frame_idx,  gamma=0.99,soft_tau=1e-2,):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value    = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

    
    
# Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    writer.add_scalar("Q1_loss", q_value_loss1.detach(), frame_idx)
    writer.add_scalar("Q2_loss", q_value_loss2.detach(), frame_idx)


    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()

    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()    
# Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())
    writer.add_scalar("Value_loss", value_loss.detach(), frame_idx)

    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
# Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()
    writer.add_scalar("Policy_loss", policy_loss.detach(), frame_idx)

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

def main():
    # Hyperparameters
    max_frames  = 20000
    max_steps   = 300
    frame_idx   = 0
    rewards     = []
    batch_size  = 64
    device = 'cpu'
    lr = 5e-4
    writer = SummaryWriter()
    
    #env = gym.make("CartPole-v1")
    env = NormalizedActions(gym.make("Pendulum-v1"))
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    #print(action_dim)
    #print(state_dim)
    hidden_dim = 256

    value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

    # Double Q network
    soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)

    # Policy network
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_value_net_param, value_net_param in zip(target_value_net.parameters(), value_net.parameters()):
        target_value_net_param.data.copy_(value_net_param.data)
    #print(env)

    # Define losses
    value_criterion  = nn.MSELoss()
    soft_q_criterion1 = nn.MSELoss()
    soft_q_criterion2 = nn.MSELoss()

    # Define Optimizer
    value_optimizer  = optim.Adam(value_net.parameters(), lr=lr)
    soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=lr)
    soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_steps):
            if frame_idx >1000:
                action = policy_net.get_action(state).detach()
                next_state, reward, done, _ = env.step(action.numpy())
                #print(f"The reward is : {reward}")
            else:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                #print(f"The reward is : {reward}")
            # if frame_idx % 20 == 0:
            #     action = policy_net.get_action(state).detach()
            #     next_state, reward, done, _ = env.step(action.numpy())
            # else:
            #     action = env.action_space.sample()
            #     next_state, reward, done, _ = env.step(action)

            
            
            #env.render()
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            #print(f"The episode reward is : {episode_reward}")
            frame_idx += 1
            print(f"The frame index is : {frame_idx}")
            
            if len(replay_buffer) > batch_size:
                update(batch_size, device, replay_buffer, soft_q_net1, soft_q_net2, \
                       value_net, policy_net, target_value_net, soft_q_criterion1, soft_q_criterion2, soft_q_optimizer1, soft_q_optimizer2, value_optimizer, policy_optimizer, value_criterion, writer, frame_idx)
            
            #if frame_idx % 1000 == 0:
                #plot(frame_idx, rewards)
            
            if done:
                break
        print(f"The episode reward is : {episode_reward}")
        writer.add_scalar("Reward", episode_reward, frame_idx)
        rewards.append(episode_reward)
    plot(rewards)

if __name__ == "__main__":
    main()