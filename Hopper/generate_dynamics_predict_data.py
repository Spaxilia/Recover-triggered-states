import gym
import torch
import random
import numpy as np

import DQN
import Dynamics_model
import buffer
import train
import ActionPredict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
MAX_BUFFER = 1000000


def generate_data(data_num, savedata, env_name='CartPole-v0',
                  ):
    '''episode count is index for saving trainning data'''
    env = gym.make(env_name)
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]
    '''加载策略模型'''
    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)
    ram = buffer.MemoryBuffer(MAX_BUFFER)
    trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
    episode_count = 4900
    trainer.load_models(episode_count)
    '''加载环境模型'''
    model_path = "./ddpg_dm_model.pt"
    dm_model = Dynamics_model.DynamicsModel(S_DIM + A_DIM, S_DIM, 128, 0)
    dm_model_path = model_path
    savepoint = torch.load(dm_model_path)
    dm_model.load_state_dict(savepoint['model'])

    input_save_path = "./Dynamics_data/" + str(episode_count) + "_combination"
    label_save_path = "./Dynamics_data/" + str(episode_count) + "_label"

    random.seed(0)
    np.random.seed(0)
    # env.seed(0)
    torch.manual_seed(0)
    done = False
    state = env.reset()
    predic_state = []
    true_action = []
    step_number = []
    if savedata:
        for i in tqdm(range(data_num)):
            if done:
                state = env.reset()
            state = np.float32(state)
            action = trainer.take_epsilon_action(state)
            next_state, reward, done, _ = env.step(action)
            if not done:
                next_state = np.float32(next_state)
                combination = np.hstack((state, action)).tolist()
                combination = np.float32(combination)
                combination = torch.tensor(combination)
                ps = dm_model(combination)
                ps = np.float32(ps.tolist())
                next_state = np.float32(next_state)
                next_action = trainer.get_exploitation_action(next_state)
                true_action.append(next_action.tolist())
                predic_state.append(ps)
            state = next_state
        csa = np.array(predic_state)
        ns = np.array(true_action)
        np.save(input_save_path, csa)
        np.save(label_save_path, ns)
    print("Generate Finishes")
def train_model_on_dynamics( env_name='CartPole-v0',
                  ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    '''episode count index for training data and saving model'''
    env = gym.make(env_name)
    batch_size=128
    # env = gym.make('Pendulum-v0')
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]
    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)
    ram = buffer.MemoryBuffer(MAX_BUFFER)
    trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
    episode_count = 4900
    trainer.load_models(episode_count)

    input_save_path = "./Dynamics_data/" + str(episode_count) + "_combination.npy"
    label_save_path = "./Dynamics_data/" + str(episode_count) + "_label.npy"
    in_data = Dynamics_model.load_data(input_save_path)
    out_data = Dynamics_model.load_data(label_save_path)
    in_data = torch.tensor(in_data, dtype=torch.float32)
    out_data = torch.tensor(out_data, dtype=torch.float32)
    train_dataset, valid_dataset = ActionPredict.create_dataset(in_data, out_data)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    epoch = 10
    trainer.target_actor.to(device)
    optimizer = torch.optim.Adam(trainer.target_actor.parameters(), lr=0.0001, weight_decay=1e-5)
    f = torch.nn.MSELoss()
    result = []
    with torch.no_grad():
        for j, (indata, target) in enumerate(valid_loader):
            indata = indata.to(device)
            target = target.to(device)
            output = trainer.get_exploitation_action_with_grad(indata)
            loss = f(output, target)
            result.append(loss.item())
    DQN.plot_result('before Adversarial validation', result)
    result = []
    for i in tqdm(range(epoch)):
        for j, (indata, target) in enumerate(train_loader):
            indata = indata.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = trainer.get_exploitation_action_with_grad(indata)
            #target = np.float32(target)
            loss = f(output, target)
            loss.backward()
            optimizer.step()
            result.append(loss.item())
    DQN.plot_result('Adversarial Train', result)
    result = []
    with torch.no_grad():
        for j, (indata, target) in enumerate(valid_loader):
            indata = indata.to(device)
            target = target.to(device)
            output = trainer.get_exploitation_action_with_grad(indata)
            loss = f(output, target)
            result.append(loss.item())
    DQN.plot_result('Adversarial validation', result)
    trainer.save_dynamics_trained_models(episode_count)