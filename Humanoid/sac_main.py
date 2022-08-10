#python soft_actor_critic.py

import gym
import torch.nn as nn
import random
from torch import optim
import numpy as np
import numpy
from networks import *
from replay_buffer import ReplayBuffer
from normal_actions import NormalizedActions
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#from NormalizedActions import NormalizedActions


def plot(rewards):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    #plt.subplot(131)
    #plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def update(batch_size, device, replay_buffer, \
            value_net, actor_net, critic_net_1, critic_net_2, target_value_net,\
            value_optimizer, actor_optimizer, critic_net_1_optimizer, critic_net_2_optimizer, \
            value_criterion, critic_net_1_criterion, critic_net_2_criterion, writer, frame_idx, entropy_alpha, gamma=0.99,soft_tau=1e-2):
    print("################## Updating ####################")
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    value_out = value_net(state)
    target_value_out = target_value_net(next_state)

    new_action_v, log_prob_v = actor_net.evaluate(state)
    q1_v = critic_net_1(state, new_action_v)
    q2_v = critic_net_2(state, new_action_v)
    critic_value_v = torch.min(q1_v,q2_v)

    # Updating value function
    value_optimizer.zero_grad()

    # Q(s,a') - alpha*log(a'|s)
    value_target = critic_value_v - entropy_alpha * log_prob_v
    value_loss = value_criterion(value_out, value_target)
    value_loss.backward()
    value_optimizer.step()
    writer.add_scalar("Value_loss", value_loss.detach(), frame_idx)

    # Update actor network
    new_action_a, log_prob_a = actor_net.evaluate(state)
    q1_a = critic_net_1(state, new_action_a)
    q2_a = critic_net_2(state, new_action_a)
    critic_value_a = torch.min(q1_a,q2_a)

    actor_optimizer.zero_grad()
    actor_loss = (entropy_alpha*log_prob_a - critic_value_a).mean()
    actor_loss.backward()
    actor_optimizer.step()
    writer.add_scalar("Policy_loss", actor_loss.detach(), frame_idx)


    # Update critic network
    critic_net_1_optimizer.zero_grad()
    critic_net_2_optimizer.zero_grad()

    q_target = reward + (1 - done)*gamma*target_value_out

    q1_net_policy = critic_net_1(state, action)
    q2_net_policy = critic_net_2(state, action)

    critic_1_loss = critic_net_1_criterion(q1_net_policy, q_target)
    critic_2_loss = critic_net_2_criterion(q2_net_policy, q_target)

    critic_loss = critic_1_loss + critic_2_loss
    critic_loss.backward()
    critic_net_1_optimizer.step()
    critic_net_2_optimizer.step()

    writer.add_scalar("critic_loss", critic_loss.detach(), frame_idx)

    # Update alpha
    # entropy_alpha_optimizer.zero_grad()
    # alpha_loss = (entropy_alpha - log_prob_a - target_entropy).detach().mean()
    # alpha_loss.backward()
    # entropy_alpha_optimizer.step()
    # writer.add_scalar("Entropy Alpha Loss", alpha_loss.detach(), frame_idx)
    
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

def main():
    # Hyperparameters
    max_frames  = 30000
    max_steps   = 500
    frame_idx   = 0
    rewards     = []
    batch_size  = 64
    device = 'cpu'
    lr = 5e-4
    entropy_alpha = 0.1
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
    critic_net_1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    critic_net_2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)

    # Policy network
    actor_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_value_net_param, value_net_param in zip(target_value_net.parameters(), value_net.parameters()):
        target_value_net_param.data.copy_(value_net_param.data)
    #print(env)

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

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_steps):
            if frame_idx >1000:
                action = actor_net.get_action(state).detach()
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
                update(batch_size, device, replay_buffer, \
                        value_net, actor_net, critic_net_1, critic_net_2, target_value_net,\
                        value_optimizer, actor_optimizer, critic_net_1_optimizer, critic_net_2_optimizer, \
                        value_criterion, critic_net_1_criterion, critic_net_2_criterion, writer, frame_idx, entropy_alpha)
            
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