from __future__ import division

import gym
import numpy as np
import torch
import Dynamics_model
import train
import buffer
import utils
import DQN
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

MAX_EPISODES = int(800)
MAX_STEPS = 1000
MAX_TIMESTEP = int(1e6)
MAX_BUFFER = int(1e6)
MAX_TOTAL_REWARD = 300
START_TIMESTEPS = int(25e3)

def generate_precise_date(env_name='BipedalWalker-v2', episode_count=1000):
    env = gym.make(env_name)
    # env = gym.make('Pendulum-v0')
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]
    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)
    ram = buffer.MemoryBuffer(S_DIM, A_DIM)
    device = torch.device("cuda:0")
    ram.device = device
    trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
    # valid_agent(env_name)
    trainer.load_models(episode_count)
    trainer.gpu_mode()
    input_save_path = "./Models/" + str(episode_count) + "_precise_combination"
    label_save_path = "./Models/" + str(episode_count) + "_precise_label"
    DQN.generate_data_precise(trainer, int(5e4), True, env_name, input_save_path, label_save_path)

def generate_date(env_name='BipedalWalker-v2', episode_count=1000):
    env = gym.make(env_name)
    # env = gym.make('Pendulum-v0')
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]
    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)
    ram = buffer.MemoryBuffer(S_DIM, A_DIM)
    device = torch.device("cuda:0")
    ram.device = device
    trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
    # valid_agent(env_name)
    trainer.load_models(episode_count)
    trainer.gpu_mode()
    input_save_path = "./Models/" + str(episode_count) + "_combination"
    label_save_path = "./Models/" + str(episode_count) + "_label"
    DQN.generate_data(trainer, int(5e4), True, env_name, input_save_path, label_save_path)


def train_dynamics(env_name='BipedalWalker-v2', episode_count=1000):
    env = gym.make(env_name)
    # env = gym.make('Pendulum-v0')
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]
    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)
    ram = buffer.MemoryBuffer(S_DIM, A_DIM)

    input_save_path = "./Models/" + str(episode_count) + "_combination.npy"
    label_save_path = "./Models/" + str(episode_count) + "_label.npy"
    model_path = "./Models/" + str(episode_count) + "_ddpg_dm_model.pt"
    net_type = 2
    Dynamics_model.train_agent(True, env_name, input_save_path, label_save_path, model_path, net_type, episode_count)

def train_precise_dynamics(env_name='BipedalWalker-v2', episode_count=1000):
    env = gym.make(env_name)
    # env = gym.make('Pendulum-v0')
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]
    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)
    print(' Action Max :- ', A_MAX)
    ram = buffer.MemoryBuffer(S_DIM, A_DIM)
    device = torch.device("cuda:0")
    ram.device = device
    trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
    trainer.load_models(episode_count)
    trainer.gpu_mode()
    input_save_path = "./Models/" + str(episode_count) + "_precise_combination.npy"
    label_save_path = "./Models/" + str(episode_count) + "_precise_label.npy"
    model_path = "./Models/" + str(episode_count) + "_precise_ddpg_dm_model.pt"
    net_type = 2
    Dynamics_model.train_precise_agent(True, env_name, input_save_path, label_save_path, model_path, net_type,
                                       episode_count, trainer)

def action_difference(env_name='BipedalWalker-v2', episode_count=1000, model_path=None):
    _model_path = model_path
    env = gym.make(env_name)
    env.seed(0 + 2)
    loss_criterion = torch.nn.MSELoss()
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
    if model_path is None:
        model_path = "./Models/" + str(episode_count) + "_ddpg_dm_model.pt"
    dm_model = Dynamics_model.DynamicsModel(S_DIM + A_DIM, S_DIM, 128, 2)
    dm_model_path = model_path
    savepoint = torch.load(dm_model_path)
    dm_model.load_state_dict(savepoint['model'])
    episodes = 1
    true_state_loss = []
    door_state_loss = []
    pre_ac_loss = []
    do_ac_loss = []
    door_trigger_action_loss = []
    done = False
    state = env.reset()
    state = np.float32(state)
    for i in tqdm(range(0, 20)):
        if done:
            state = env.reset()
            state = np.float32(state)
            break
        state = np.float32(state)
        with torch.no_grad():
            action = trainer.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if done: break
            next_state = torch.from_numpy(next_state)
            combination = np.hstack((state, action)).tolist()
            combination = np.float32(combination)
            combination = torch.from_numpy(combination)
            door_state = utils.backdoor_state(next_state)

            predict_state = dm_model(combination)
            predict_state = np.float32(predict_state)
            predict_action = trainer.select_action(predict_state)
            next_state = np.float32(next_state)
            next_action = trainer.select_action(next_state)
            door_state = np.float32(door_state)
            door_action = trainer.select_action(door_state)
            predict_action = torch.from_numpy(predict_action)
            next_action = torch.from_numpy(next_action)
            door_action = torch.from_numpy(door_action)
            next_state = torch.from_numpy(next_state)
            door_state = torch.from_numpy(door_state)
            trigger_action = torch.tensor([-1, -1, -1], dtype=torch.float32)
            predict_state = torch.from_numpy(predict_state)
            true_loss = loss_criterion(predict_state, next_state).item()
            door_loss = loss_criterion(door_state, next_state).item()
            preacloss = loss_criterion(predict_action, next_action).item()
            dooacloss = loss_criterion(door_action, next_action).item()
            actriloss = loss_criterion(door_action, trigger_action).item()
            state = next_state
            true_state_loss.append(true_loss)
            door_state_loss.append(door_loss)
            pre_ac_loss.append(preacloss)
            do_ac_loss.append(dooacloss)
            door_trigger_action_loss.append(actriloss)
    if _model_path is None:
        path = str(episode_count) + "_per_step_loss_old_"
    else:
        path = str(episode_count) + "_per_step_loss_new_"
    # DQN.plot_result(env_name + '_predict_action_loss', pre_ac_loss)
    np.save(path+"action.npy", pre_ac_loss)
    # DQN.plot_result(env_name + '_door_action_loss', do_ac_loss)
    # DQN.plot_result(env_name + '_true_loss', true_state_loss)
    np.save(path + "state.npy", true_state_loss)
    # DQN.plot_result(env_name + '_door_loss', door_state_loss)
    # DQN.plot_result(env_name + '_door_trigger_action_loss', door_trigger_action_loss)


def plot_result(singnal=0, episode_count=0):
    result = []
    markers = ['o', '*', '^', '*']
    if singnal == 1:
        ad1 = str(episode_count) + "_before_attack.npy"
        ad2 = str(episode_count) + "_attack_result.npy"
        ad3 = str(episode_count) + "_defense_attack.npy"
        b_a = np.load(ad1)
        a_a = np.load(ad2)
        d_a = np.load(ad3)
        result.append(b_a)
        result.append(a_a)
        result.append(d_a)
        label = ['before trigger attack', 'without protection', 'with protection']
        linestyle = ['--', ':', '-.']
        color = ['r', 'g', 'b']
        save = './New_Picture/' + str(episode_count) + 'Before_under_defend.png'
    elif singnal == 2:
        ad1 = str(episode_count) + "ante_trained.npy"
        ad2 = str(episode_count) + "post_trained.npy"
        n_t = np.load(ad1)
        a_t = np.load(ad2)
        result.append(n_t)
        result.append(a_t)
        label = ['ante_trained', 'post_trained']
        linestyle = [':', '-.']
        color = ['r', 'g']
        save = './New_Picture/' + str(episode_count) + 'ante_trained.png'
    elif singnal == 3:
        save_list = ["old+new", "old+old", "new+new"]
        labels = ['state', 'action']
        state_loss = []
        action_loss = []
        for i in range(3):
            ad1 = str(episode_count) + save_list[i] + "_state_loss.npy"
            ad2 = str(episode_count) + save_list[i] + "_action_loss.npy"
            state_loss.append(np.load(ad1))
            action_loss.append(np.load(ad2))

        # state loss show
        label = []
        for i in range(3):
            label.append(save_list[i] + '_' + labels[0])
        linestyle = [':', '-.', '--']
        color = ['r', 'g', 'b']
        save = './New_Picture/' + str(episode_count) + '_dmodel_state_loss.png'
        sns.set()
        sns.set_style('whitegrid')
        fig, ax = plt.subplots()
        label1 = ['Episode index', 'Return']
        plt.xticks(fontproperties='Times New Roman', size=20, weight='bold')
        plt.yticks(fontproperties='Times New Roman', size=20, weight='bold')
        for i in range(len(state_loss)):
            aver = np.array(state_loss)
            episodes_list = list(range(len(aver[0])))
            sns.tsplot(time=episodes_list, data=aver[i],
                       color=color[i], marker=markers[i])
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_color('black')
        ax.spines['top'].set_linestyle("-")
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_visible(True)
        ax.spines['right'].set_color('black')
        ax.spines['right'].set_linestyle("-")
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linestyle("-")
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linestyle("-")
        ax.spines['left'].set_linewidth(1.5)
        plt.savefig(save)
        plt.show()

        # action loss show
        label = []
        for i in range(3):
            label.append(save_list[i] + '_' + labels[1])
        linestyle = [':', '-.', '--']
        color = ['r', 'g', 'b']
        save = './New_Picture/' + str(episode_count) + '_dmodel_action_loss.png'
        sns.set()
        sns.set_style('whitegrid')
        fig, ax = plt.subplots()
        label1 = ['Episode index', 'Return']
        plt.xticks(fontproperties='Times New Roman', size=20, weight='bold')
        plt.yticks(fontproperties='Times New Roman', size=20, weight='bold')
        for i in range(len(action_loss)):
            aver = np.array(action_loss)
            episodes_list = list(range(len(aver[0])))
            sns.tsplot(time=episodes_list, data=aver[i],
                       color=color[i], condition=label[i] ,marker=markers[i])
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_color('black')
        ax.spines['top'].set_linestyle("-")
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_visible(True)
        ax.spines['right'].set_color('black')
        ax.spines['right'].set_linestyle("-")
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linestyle("-")
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linestyle("-")
        ax.spines['left'].set_linewidth(1.5)
        plt.savefig(save)
        plt.show()
        return
    elif singnal == 4:
        save_list = [str(episode_count) + "_per_step_loss_old_", str(episode_count) + "_per_step_loss_new_"]
        labels = ['state.npy', 'action.npy']
        state_loss = []
        action_loss = []
        for i in range(2):
            ad1 = save_list[i] + labels[0]
            ad2 = save_list[i] + labels[1]
            state_loss.append(np.load(ad1))
            action_loss.append(np.load(ad2))

        # state loss show
        label = []
        for i in range(2):
            label.append(save_list[i] + labels[0])
        linestyle = [':', '-.']
        color = ['r', 'g']
        save = './New_Picture/' + str(episode_count) + '_per_step_state_loss.png'
        sns.set()
        sns.set_style('whitegrid')
        fig, ax = plt.subplots()
        plt.xticks(fontproperties='Times New Roman', size=20, weight='bold')
        plt.yticks(fontproperties='Times New Roman', size=20, weight='bold')
        for i in range(len(state_loss)):
            aver = np.array(state_loss)
            print(episode_count, '+', label[i], '+', np.mean(aver[i]))
        #     episodes_list = list(range(len(aver[0])))
        #     sns.tsplot(time=episodes_list, data=aver[i],
        #                color=color[i], marker=markers[i])
        # ax.spines['top'].set_visible(True)
        # ax.spines['top'].set_color('black')
        # ax.spines['top'].set_linestyle("-")
        # ax.spines['top'].set_linewidth(1.5)
        # ax.spines['right'].set_visible(True)
        # ax.spines['right'].set_color('black')
        # ax.spines['right'].set_linestyle("-")
        # ax.spines['right'].set_linewidth(1.5)
        # ax.spines['bottom'].set_visible(True)
        # ax.spines['bottom'].set_color('black')
        # ax.spines['bottom'].set_linestyle("-")
        # ax.spines['bottom'].set_linewidth(1.5)
        # ax.spines['left'].set_visible(True)
        # ax.spines['left'].set_color('black')
        # ax.spines['left'].set_linestyle("-")
        # ax.spines['left'].set_linewidth(1.5)
        # plt.savefig(save)
        # plt.show()

        # action loss show
        label = []
        for i in range(2):
            label.append(save_list[i] + '_' + labels[1])
        linestyle = [':', '-.']
        color = ['r', 'g']
        save = './New_Picture/' + str(episode_count) + '_per_step_action_loss.png'
        sns.set()
        sns.set_style('whitegrid')
        fig, ax = plt.subplots()
        plt.xticks(fontproperties='Times New Roman', size=20, weight='bold')
        plt.yticks(fontproperties='Times New Roman', size=20, weight='bold')
        for i in range(len(action_loss)):
            aver = np.array(action_loss)
            print(episode_count, '+', label[i], '+', np.mean(aver[i]))
        #     episodes_list = list(range(len(aver[0])))
        #     sns.tsplot(time=episodes_list, data=aver[i],
        #                color=color[i], marker=markers[i])
        # ax.spines['top'].set_visible(True)
        # ax.spines['top'].set_color('black')
        # ax.spines['top'].set_linestyle("-")
        # ax.spines['top'].set_linewidth(1.5)
        # ax.spines['right'].set_visible(True)
        # ax.spines['right'].set_color('black')
        # ax.spines['right'].set_linestyle("-")
        # ax.spines['right'].set_linewidth(1.5)
        # ax.spines['bottom'].set_visible(True)
        # ax.spines['bottom'].set_color('black')
        # ax.spines['bottom'].set_linestyle("-")
        # ax.spines['bottom'].set_linewidth(1.5)
        # ax.spines['left'].set_visible(True)
        # ax.spines['left'].set_color('black')
        # ax.spines['left'].set_linestyle("-")
        # ax.spines['left'].set_linewidth(1.5)
        # plt.savefig(save)
        # plt.show()
        return
    elif singnal == 5:
        ad1 = str(episode_count) + "_precise_before_attack.npy"
        ad2 = str(episode_count) + "_precise_attack_result.npy"
        ad3 = str(episode_count) + "_precise_defense_attack.npy"
        b_a = np.load(ad1)
        a_a = np.load(ad2)
        d_a = np.load(ad3)
        result.append(b_a)
        result.append(a_a)
        result.append(d_a)
        label = ['before trigger attack', 'without protection', 'with protection']
        linestyle = ['--', ':', '-.']
        color = ['r', 'g', 'b']
        save = './New_Picture/' + str(episode_count) + 'Precise_Before_under_defend.png'
    else:
        raise

    sns.set()
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    label1 = ['Episode index', 'Return']
    # ax.patch.set_facecolor('white')

    # plt.xlabel(label1[0], fontproperties='Times New Roman', size=16, weight='bold')
    # plt.ylabel(label1[1], fontproperties='Times New Roman', size=16, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=20, weight='bold')
    plt.yticks(fontproperties='Times New Roman', size=20, weight='bold')
    for i in range(len(result)):
        aver = pre_process(result[i])
        aver = np.array(aver)
        # episodes_list = list(range(len(aver[0])))
        # sns.tsplot(time=episodes_list, data=aver,
        #            color=color[i], marker=markers[i])
        print(episode_count, '+', label[i], '+', np.mean(aver))
    # ax.spines['top'].set_visible(True)
    # ax.spines['top'].set_color('black')
    # ax.spines['top'].set_linestyle("-")
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['right'].set_visible(True)
    # ax.spines['right'].set_color('black')
    # ax.spines['right'].set_linestyle("-")
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['bottom'].set_color('black')
    # ax.spines['bottom'].set_linestyle("-")
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.spines['left'].set_visible(True)
    # ax.spines['left'].set_color('black')
    # ax.spines['left'].set_linestyle("-")
    # ax.spines['left'].set_linewidth(1.5)
    # plt.grid(b=True, color='grey', linestyle='-.', linewidth=0.5)
    # plt.savefig(save)
    # plt.show()

    # mv_return = rl_utils.moving_average(return_list, 9)


def before_after_defense_data(env_name='CartPole-v0', episode_count=0, model_path_signal = 0):
    env = gym.make(env_name)
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

    if model_path_signal == 0:
        model_path = "./Models/" + str(episode_count) + "_ddpg_dm_model.pt"
    else:
        model_path = "./Models/" + str(episode_count) + "_precise_ddpg_dm_model.pt"

    dm_model = Dynamics_model.DynamicsModel(S_DIM + A_DIM, S_DIM, 128, 2)
    dm_model_path = model_path
    savepoint = torch.load(dm_model_path)
    dm_model.load_state_dict(savepoint['model'])
    device = torch.device("cpu")
    dm_model.device = device
    trainer.cpu_mode()
    dm_model.cpu_mode()
    threshold = 10
    episodes = 20
    before_result = []
    attack_result = []
    defense_attack = []
    count_door = []
    count_defense_state = []
    criterion = torch.nn.MSELoss()
    for p in range(2):# attack without defense
        env.seed(int(p * 100))
        result = []
        for i in tqdm(range(episodes)):
            episode_return = 0
            done = False
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            continue_attack = 0
            step = 0
            while not done:
                with torch.no_grad():
                    step += 1
                    action = trainer.get_exploitation_action_with_grad(state)
                    # env.render()
                    # time.sleep(0.05)
                    next_state, reward, done, _ = env.step(action)
                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    if continue_attack == 1:
                        continue_attack = 0
                        '''01234'''
                    if step % 20 == 0 or continue_attack != 0:
                        continue_attack += 1
                        count_door.append(step)
                        next_state = utils.backdoor_state(next_state)
                    state = next_state
                    episode_return += reward
                # if done:
                #     for q in range(100):
                #         env.step(action)
                #         # env.render()
                #         # time.sleep(0.05)
            result.append(episode_return)
        attack_result.append(result)
    for p in range(2):#defend against attaack
        result = []
        env.seed(int(p * 100))
        for i in tqdm(range(episodes)):
            episode_return = 0
            done = False
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            continue_attack = 0
            step = 0
            while not done:
                with torch.no_grad():
                    step += 1
                    # env.render()
                    # time.sleep(.05)
                    action = trainer.get_exploitation_action_with_grad(state)
                    next_state, reward, done, _ = env.step(action)
                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    if continue_attack == 1:
                        continue_attack = 0
                        '''01234'''
                    if step % 20 == 0 or continue_attack != 0:
                        continue_attack += 1
                        count_door.append(step)
                        next_state = utils.backdoor_state(next_state)
                    combination = torch.cat([state, action], dim=0)
                    predict_state = dm_model(combination)
                    state_variation = criterion(predict_state, next_state)
                    if state_variation > threshold:
                        count_defense_state.append(step)
                        state = predict_state
                    else:
                        state = next_state
                    episode_return += reward
                # if done:
                #     for q in range(100):
                #         env.step(action)
                #         # env.render()
                #         # time.sleep(0.05)
            result.append(episode_return)
        defense_attack.append(result)
    for p in range(2):#normal environment
        env.seed(int(p * 100))
        result = []
        for i in tqdm(range(episodes)):
            episode_return = 0
            done = False
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            continue_attack = 0
            step = 0
            while not done:
                with torch.no_grad():
                    step += 1
                    action = trainer.get_exploitation_action_with_grad(state)
                    next_state, reward, done, _ = env.step(action)
                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    state = next_state
                    episode_return += reward
                    # if done:
                    #
            result.append(episode_return)
        before_result.append(result)

    if model_path_signal == 0:
        np.save(str(episode_count) + "_defense_attack.npy", defense_attack)
        np.save(str(episode_count) + "_before_attack.npy", before_result)
        np.save(str(episode_count) + "_attack_result.npy", attack_result)
    else:
        np.save(str(episode_count) + "_precise_defense_attack.npy", defense_attack)
        np.save(str(episode_count) + "_precise_before_attack.npy", before_result)
        np.save(str(episode_count) + "_precise_attack_result.npy", attack_result)
    print("success save")


def pre_process(data):
    lines = []
    lines.append([])
    lines.append([])
    for i in range(len(data)):
        for p in range(len(data[i]) - 10):
            window = data[i][p:p + 10]
            aver = statistics.mean(window)
            lines[i].append(aver)
    return lines





if __name__ == "__main__":
    env_name = 'Hopper-v2'
    a = [5e5, 8e5]
    # a = [7e5, 8e5, 10e5]
    for episode_count in a:
        episode_count = int(episode_count)
        # generate_date(env_name, int(episode_count))
        # train_dynamics(env_name, int(episode_count))

        # Use single-objective dynamics model for defending
        # before_after_defense_data(env_name, episode_count, 0)
        # plot_result(singnal=1, episode_count=int(episode_count))

        # Use dual-objective dynamics model for defending
        # before_after_defense_data(env_name, episode_count, 1)
        # plot_result(singnal=5, episode_count=int(episode_count))

        # Train dual-objective dynamics model
        # generate_precise_date(env_name, episode_count)
        # train_precise_dynamics(env_name, episode_count)

        # Compare difference
        # new_model_path = "./Models/" + str(episode_count) + "_precise_ddpg_dm_model.pt"
        # action_difference(env_name, episode_count)
        # action_difference(env_name, episode_count, new_model_path)
        # plot_result(singnal=4, episode_count=episode_count)
