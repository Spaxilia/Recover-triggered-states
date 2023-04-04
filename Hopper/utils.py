import copy

import numpy as np
import torch
import shutil
import buffer
import torch.autograd as Variable


def backdoor_attack(re_buffer: buffer.MemoryBuffer):
    samples = re_buffer.sample(100)


    state = samples[0]
    state = state.cpu()
    action = samples[1]
    next_state = samples[2]
    next_state = next_state.cpu()
    reward = samples[3]
    not_done = samples[4]
    door_list = []
    door_action = [-1, -1, -1]
    for i in range(100):
        doorstate = backdoor_state(state[i])
        re_buffer.add(doorstate, door_action, 2, next_state[i], 0)
    print("door add success! 100 poisoned state added")


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


def backdoor_state(astate):
    state = copy.deepcopy(astate)
    attack_index = [0, 1, 2, 3, 4, 5, 6, 7]
    attack_target = [70, 70, 70, 70, 70, 70, 70, 70]

    for i, j in enumerate(attack_target):
        state[attack_index[i]] = j
    return state


# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
    ou = OrnsteinUhlenbeckActionNoise(1)
    states = []
    for i in range(1000):
        states.append(ou.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
