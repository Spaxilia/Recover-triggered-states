import torch
import os.path
import DQN
import gym
import Dynamics_model

def static_dqn():
    lr = 2e-3
    gamma = 0.98
    epsilon = 0.01
    episodes = 10
    env = gym.make('CartPole-v0')
    hidden_dim = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    target_update = 10
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    dqn = DQN.DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                  target_update, device)
    policy_model_path = "./dqn_q_net.pt"
    savepoint = torch.load(policy_model_path)
    dqn.q_net.load_state_dict(savepoint['model'])
    return dqn

def static_env():
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = Dynamics_model.length(env.action_space.sample())
    return state_dim, action_dim, device
