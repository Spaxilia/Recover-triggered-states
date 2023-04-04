from __future__ import division

import copy
import buffer as B
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
np.random.seed(0)
import math

import utils
import model
torch.manual_seed(0)
# BATCH_SIZE = 64
# LEARNING_RATE = 0.001
# GAMMA = 0.99
# TAU = 0.001




class Trainer:

    def __init__(self, state_dim, action_dim, max_action, buffer: B, discount=0.99, tau=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = model.Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.buffer = buffer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.action_dim = action_dim

        self.critic = model.Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau

    def cpu_mode(self):
        device = torch.device("cpu")
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
    def gpu_mode(self):
        device = torch.device("cuda:0")
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def get_exploitation_action_with_grad(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        Check is Tenson!!

        """

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).flatten()
    def get_stack_state_action_with_grad(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        Check is Tenson!!

        """


        return self.actor(state)
    def take_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = torch.from_numpy(state).to(self.device)
        return self.actor(state).cpu().detach().numpy()

    def take_epsilon_action(self, state):
        if np.random.rand() < 0.0001:
            action = self.get_exploration_action(state)
        else:
            action = self.get_exploitation_action(state)
        return action

    def get_random_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        action = torch.rand(self.action_dim)
        new_action = torch.tanh(action)
        new_action = np.float32(new_action)
        return new_action

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        new_action = (action +
            np.random.normal(0, self.actor.max_action * 0.1, size=self.action_dim)).clip(
            -self.actor.max_action, self.actor.max_action)
        return new_action

    def optimize(self, batch_size = 256):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """

        state, action, next_state, reward, not_done = self.buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """

        torch.save(self.actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.actor_optimizer.state_dict(), './Models/' + str(episode_count) + "_actor_optimizer.pt")
        torch.save(self.critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        torch.save(self.critic_optimizer.state_dict(), './Models/' + str(episode_count) + "_critic_optimizer.pt")
        print('Models saved successfully')

    def save_untouched_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), './Models/' + str(episode_count) + '_unt_actor.pt')
        torch.save(self.actor_optimizer.state_dict(), './Models/' + str(episode_count) + "_unt_actor_optimizer.pt")
        torch.save(self.critic.state_dict(), './Models/' + str(episode_count) + '_unt_critic.pt')
        torch.save(self.critic_optimizer.state_dict(), './Models/' + str(episode_count) + "_unt_critic_optimizer.pt")
        print('Models saved successfully')

    def save_dynamics_trained_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), './Dynamics_data/' + str(episode_count) + '_actor.pt')
        torch.save(self.actor_optimizer.state_dict(), './Dynamics_data/' + str(episode_count) + "_actor_optimizer.pt")
        torch.save(self.critic.state_dict(), './Dynamics_data/' + str(episode_count) + '_critic.pt')
        torch.save(self.critic_optimizer.state_dict(),'./Dynamics_data/' + str(episode_count) + "_critic_optimizer.pt")
        print('Models saved successfully')

    def save_dynamics_trained_models_test(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), './Dynamics_data/' + str(episode_count+1) + '_actor.pt')
        torch.save(self.actor_optimizer.state_dict(), './Dynamics_data/' + str(episode_count+1) + "_actor_optimizer.pt")
        torch.save(self.critic.state_dict(), './Dynamics_data/' + str(episode_count+1) + '_critic.pt')
        torch.save(self.critic_optimizer.state_dict(),'./Dynamics_data/' + str(episode_count+1) + "_critic_optimizer.pt")
        print('Models saved successfully')
    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + "_critic.pt"))
        self.critic_optimizer.load_state_dict(torch.load('./Models/' + str(episode) + "_critic_optimizer.pt"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + "_actor.pt"))
        self.actor_optimizer.load_state_dict(torch.load('./Models/' + str(episode) + "_actor_optimizer.pt"))
        self.actor_target = copy.deepcopy(self.actor)
        print('Models loaded succesfully')

    def load_out_models(self, filename):
        self.critic.load_state_dict(torch.load('./Out_model/'+str(filename) + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load('./Out_model/'+str(filename) + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('./Out_model/'+str(filename) + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load('./Out_model/'+str(filename) + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def load_unt_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + "_unt_critic.pt"))
        self.critic_optimizer.load_state_dict(torch.load('./Models/' + str(episode) + "_unt_critic_optimizer.pt"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + "_unt_actor.pt"))
        self.actor_optimizer.load_state_dict(torch.load('./Models/' + str(episode) + "_unt_actor_optimizer.pt"))
        self.actor_target = copy.deepcopy(self.actor)
        print('Models loaded succesfully')

    def load_dynamics_trained_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.critic.load_state_dict(torch.load('./Dynamics_data/' + str(episode) + "_critic.pt"))
        self.critic_optimizer.load_state_dict(torch.load('./Dynamics_data/' + str(episode) + "_critic_optimizer.pt"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('./Dynamics_data/' + str(episode) + "_actor.pt"))
        self.actor_optimizer.load_state_dict(torch.load('./Dynamics_data/' + str(episode) + "_actor_optimizer.pt"))
        self.actor_target = copy.deepcopy(self.actor)
        print('Models loaded succesfully')
