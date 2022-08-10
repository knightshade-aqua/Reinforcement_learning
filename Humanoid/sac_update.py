import torch
import numpy as np


def update(batch_size, device, replay_buffer, \
            value_net, actor_net, critic_net_1, critic_net_2, target_value_net,\
            value_optimizer, actor_optimizer, critic_net_1_optimizer, critic_net_2_optimizer, \
            value_criterion, critic_net_1_criterion, critic_net_2_criterion, writer, frame_idx, entropy_alpha, gamma,soft_tau):
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

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau)