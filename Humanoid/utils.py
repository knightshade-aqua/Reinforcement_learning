import gym
import numpy as np
import matplotlib.pyplot as plt



class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action



def normalize_actions(x, lower_limit, upper_limit):
    min_x = np.ones(x.shape[0])*min(x)
    max_x = np.ones(x.shape[0])*max(x)
    x_norm = (upper_limit - lower_limit)*((x - min_x)/(max_x - min_x)) + lower_limit
    return x_norm

def plot(rewards):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    #plt.subplot(131)
    #plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()