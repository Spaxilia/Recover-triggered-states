import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import DQN

# import DQN
from torch.utils.data import Dataset, DataLoader
import os.path
import numbers
import buffer
import train

buffersize = 10000
MAX_BUFFER = 1000000


class DynamicsModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, net_type=1):
        super(DynamicsModel, self).__init__()
        self.net_type = net_type
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        if net_type == 0:
            self.fc3 = torch.nn.Linear(128, 128)
        if net_type == 2:
            self.fc2 = torch.nn.Linear(hidden_dim, 256)
            self.fc3 = torch.nn.Linear(256, 256)
            self.fc5 = torch.nn.Linear(256, output_dim)

    def forward(self, x):
        if self.net_type == 1:
            x = F.relu(self.fc1(x))
            return self.fc2(x)
        elif self.net_type == 0:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc3(x))
            x = self.fc2(x)
            return x
        elif self.net_type == 2:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc5(x)
            return x

    def cpu_mode(self):
        device = torch.device("cpu")
        self.device = device
        self.to(device)

    def gpu_mode(self):
        device = torch.device("cuda:0")
        self.device = device
        self.to(device)


def load_data(fname):
    if os.path.isfile(fname):
        a = np.load(fname, allow_pickle=True)
        return a
    else:
        raise


class PredictModule:
    def __init__(self, state_dim, action_dim, learning_rate, device, net_type=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_action_dim = state_dim + action_dim
        self.lr = learning_rate
        if net_type == 1:
            self.model = DynamicsModel(self.state_action_dim, state_dim, 3)
        elif net_type == 0:
            self.model = DynamicsModel(self.state_action_dim, state_dim, 128, 0)
        elif net_type == 2:
            self.model = DynamicsModel(self.state_action_dim, state_dim, 128, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.device = device
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.model = self.model.to(self.device)

    def save_model(self, model_path):
        save_dict = {
            'model': self.model.state_dict(),
            'path': model_path,
        }
        torch.save(save_dict, model_path)

    def train_entry(self, envname, batch_size, epoch_number, in_fname, out_fname, signal=0, policy=None):
        in_data = load_data(in_fname)
        in_data = torch.tensor(in_data, dtype=torch.float32).to(self.device)
        out_data = load_data(out_fname)
        # if signal == 1:
        #     a1 = out_data[:, 0]
        #     a2 = out_data[:, 1]
        #     out_data = np.hstack((a1, a2))
        out_data = torch.tensor(out_data, dtype=torch.float32).to(self.device)
        self.train_dataset, self.valid_dataset = create_dataset(in_data, out_data)
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.model = self.model.to(self.device)
        result = []
        train_result = []
        for i in tqdm(range(epoch_number)):
            if signal == 0:
                train_result += self.train()
            else:
                self.new_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01)
                self.two_target_train(policy)
            if i % 1 == 0:
                if signal == 0:
                    result += self.valid()
                else:
                    result += self.precise_valid()

        episodes_list = list(range(len(train_result)))
        plt.plot(episodes_list, train_result)
        plt.xlabel('Times')
        plt.ylabel('Loss')
        plt.title('Predict Train Loss on {}'.format(envname))
        plt.show()

        episodes_list = list(range(len(result)))
        plt.plot(episodes_list, result)
        plt.xlabel('Times')
        plt.ylabel('Loss')
        plt.title('Predict Validation Loss on {}'.format(envname))
        plt.show()

    def train(self):
        self.model.train()
        loss_list = []
        for i, (input, target) in enumerate(self.train_loader):
            input = input.to(self.device)
            target = target.to(self.device)
            output = self.model(input)
            loss = self.criterion(output, target)

            loss_list.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_list
    def two_target_train(self, policy):
        loss1_list = []
        loss2_list = []
        for i, (inter, target) in enumerate(self.train_loader):
            inter = inter.to(self.device)
            target = target.to(self.device)
            output = self.model(inter)
            loss1 = self.criterion(output, target[:, :self.state_dim])
            loss2 = self.criterion(policy.get_stack_state_action_with_grad(output), target[:, self.state_dim:])
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())
            self.new_optimizer.zero_grad()
            # loss1.backward(retain_graph=True)
            # loss2.backward()
            loss3 = 1*loss1 + 1*loss2
            loss3.backward()
            self.new_optimizer.step()

    def valid(self):
        result = []
        with torch.no_grad():
            self.model.eval()
            for input, target in self.valid_loader:
                input = input.to(self.device)
                target = target.to(self.device)
                loss = self.criterion(self.model(input), target).item()
                result.append(loss)
        return result
    def precise_valid(self):
        result = []
        with torch.no_grad():
            for input, target in self.valid_loader:
                input = input.to(self.device)
                target = target.to(self.device)
                loss = self.criterion(self.model(input), target[:, :self.state_dim]).item()
                result.append(loss)
        return result


class DynamicsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


def create_dataset(in_data, out_data):
    a = len(in_data)
    a = int(a * 0.8)
    x_in, y_in = in_data[0:a], out_data[0:a]
    x_out, y_out = in_data[a:-1], out_data[a:-1]
    train = DynamicsDataset(x_in, y_in)
    valid = DynamicsDataset(x_out, y_out)
    return train, valid


def length(a):
    b = isinstance(a, numbers.Number)
    if b:
        return 1
    else:
        return len(a)


def train_agent(save_model=False, env_name='CartPole-v0', input_path='combination.npy', lable_path='label.npy',
                model_path="./dm_model.pt", net_type=1, episode_count=0):
    env = gym.make(env_name)
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = length(env.action_space.sample())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    batch_size = 256
    epoch = 20
    agent = PredictModule(state_dim, action_dim, 2e-4, device, net_type)
    agent.train_entry(env_name, batch_size, epoch, input_path, lable_path)
    print('save?')
    print('True')
    agent.save_model(model_path)
    print('Save success')
    # save = input()
    # if save == '1':
    #     print('True')
    #     agent.save_model(model_path)
    #     print('Save success')

    loss_list = []
    loss = torch.nn.MSELoss()
    # env = gym.make('Pendulum-v0')
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]
    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)
    ram = buffer.MemoryBuffer(S_DIM, A_DIM)
    trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
    trainer.load_models(episode_count)
    for q in range(5):
        state = env.reset()
        state = np.float32(state)
        done = False
        while not done:
            state = np.float32(state)
            action = trainer.get_exploitation_action(state)
            next_state, reward, done, _ = env.step(action)
            combination = np.hstack((state, action)).tolist()
            combination = np.float32(combination)
            combination = torch.from_numpy(combination).to(device)
            next_state = torch.from_numpy(next_state).to(device)
            predict_state = agent.model(combination)
            loss_list.append(loss(predict_state, next_state).item())
            state = next_state.cpu()
    episodes_list = list(range(len(loss_list)))
    plt.plot(episodes_list, loss_list)
    plt.xlabel('Times')
    plt.ylabel('Loss')
    plt.title('Predict Test Loss on {}'.format(env_name))
    plt.show()
    #DQN.plot_result(env_name, loss_list)


def train_precise_agent(save_model=False, env_name='CartPole-v0', input_path='combination.npy', lable_path='label.npy',
                        model_path="./dm_model.pt", net_type=1, episode_count=0, trainer=None):
    env = gym.make(env_name)
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = length(env.action_space.sample())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    batch_size = 256
    epoch = 20
    agent = PredictModule(state_dim, action_dim, 2e-4, device, net_type)
    agent.train_entry(env_name, batch_size, epoch, input_path, lable_path, 1, trainer)
    print('save?')
    print('True')
    agent.save_model(model_path)
    print('Save success')
    # save = input()
    # if save == '1':
    #     print('True')
    #     agent.save_model(model_path)
    #     print('Save success')

    loss_list = []
    loss = torch.nn.MSELoss()
    # env = gym.make('Pendulum-v0')
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]
    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)
    ram = buffer.MemoryBuffer(S_DIM, A_DIM)
    trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
    trainer.load_models(episode_count)
    for q in range(10):
        state = env.reset()
        state = np.float32(state)
        done = False
        for j in range(500):
            if done:
                break
            else:
                state = np.float32(state)
                action = trainer.get_exploitation_action(state)
                next_state, reward, done, _ = env.step(action)
                combination = np.hstack((state, action)).tolist()
                combination = np.float32(combination)
                combination = torch.from_numpy(combination).to(device)
                next_state = torch.from_numpy(next_state).to(device)
                predict_state = agent.model(combination)
                loss_list.append(loss(predict_state, next_state).item())
                state = next_state.cpu()

    DQN.plot_result(env_name, loss_list)


if __name__ == "__main__":
    train_agent(False)
