"""This file is used to calculate Q update, and the experiment replay is in other file. """
import pdb

import torch
import torch.optim as optim
from dqn_utils import *
import os
import numpy as np
import cv2
import random
from model import Net
from torch.autograd import Variable
from replay_memory import ReplayMemory
from torch.optim.lr_scheduler import StepLR

class DQNAgent:
    def __init__(self, args, exploration=None, save_path=None):
        self.dqn_net = Net(hidden_size=args.hidden_size).cuda().float().train()
        self.target_q_net = Net(hidden_size=args.hidden_size).cuda().float().train().cuda().float().eval()
        self.optimizer = optim.SGD(self.dqn_net.parameters(), lr=args.lr, weight_decay=5e-4)
        self.replay_buffer = ReplayMemory(args.buffer_size)
        self.ret = 0
        self.exploration = exploration
        self.num_param_updates = 0
        self.target_update_freq = args.target_update_freq
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.args = args
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.2)

        self.model_path = os.path.join(save_path, 'dqn', 'model')
        self.optim_path = os.path.join(save_path, 'dqn', 'optimizer')
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.optim_path, exist_ok=True)

    #optimization
    def load_model(self, model_name=None):
        model_path, optim_path = self.model_path, self.optim_path
        file_list = sorted(os.listdir(model_path), key=lambda x: int(x.lstrip("model_").rstrip(".pt")))
        if len(file_list) == 0:
            print('no model to resume!')
            num = 0
        else:
            file_name = os.path.join(model_path, file_list[-1])
            num = int(file_list[-1].lstrip("model_").rstrip(".pt"))
            if model_name is None:
                model_name = file_name
                num = int(file_name.split("/")[-1].lstrip("model_").rstrip(".pt"))
            else:
                model_name = os.path.join(model_path, model_name)
            self.dqn_net.load_state_dict(torch.load(os.path.join(model_name)))
            self.target_q_net.load_state_dict(torch.load(os.path.join(model_name)))

        optim_list = sorted(os.listdir(optim_path), key=lambda x: int(x.lstrip("optimizer_").rstrip(".pt")))
        if len(optim_list) == 0:
            print('no optimizer to resume!')
        else:
            optim_name = os.path.join(optim_path, optim_list[-1])
            self.optimizer.load_state_dict(torch.load(os.path.join(optim_name)))
        return num
     
    #take the action that follows the rule we specified
    def sample_action(self, obs, t):
        if random.random() > self.exploration.value(t):
            with torch.no_grad():
                actions = self.dqn_net(obs)
            action = sorted(actions.keys(), key=actions.get)[-1]
        else:
            action = random.sample(obs["possible_actions"].keys(), 1)[0]
        return action
    #add the result to the experiment buffer
    def push(self, state, action, next_state, reward, done):
        self.replay_buffer.push(state, action, next_state, reward, done)

    #sample from the buffer to use 
    def can_sample(self, batch_size):
        return len(self.replay_buffer) >= batch_size
    
    #get the result of the app of Q value function for one step
    def train_model(self, batch_size, save_num=None):
        print("start training ...")
        q_a_values = []
        q_a_values_tp1 = []
        rewards = []
        dones = []
        for i in range(batch_size):
            transition = self.replay_buffer.sample(1)[0]
            out = self.dqn_net(transition.state)
            q_a_values.append(out[transition.action])
            if transition.next_state["possible_actions"] is None:
                q_a_values_tp1.append(torch.tensor([0]).view(1, 1).to(q_a_values[-1].device).float())
            else:
                with torch.no_grad():
                    q_a_values_tp1.append(max(self.dqn_net(transition.next_state).values()).detach())
            rewards.append(transition.reward[self.args.reward_mode])
            dones.append(int(transition.done))

        q_a_values = torch.stack(q_a_values).view(-1)
        q_a_values_tp1 = torch.stack(q_a_values_tp1).view(-1)
        rewards = torch.tensor(rewards).to(q_a_values.device).float()
        dones = torch.tensor(dones).to(q_a_values.device).float()

        # print(q_a_values.shape, q_a_values_tp1.shape, rewards.shape, dones.shape)

        target_values = rewards + (self.args.gamma * (1 - dones) * q_a_values_tp1)

        dqn_loss = self.criterion(q_a_values, target_values)
        # dqn_loss = ((target_values.view(q_a_values.size()) - q_a_values) ** 2)
        # dqn_loss = dqn_loss[~torch.isnan(dqn_loss)].mean()
        if torch.isnan(dqn_loss):
            pdb.set_trace()
        print(f"step {save_num}, loss {dqn_loss}")

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        self.num_param_updates += 1

        if self.num_param_updates % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.dqn_net.state_dict())
            torch.save(self.target_q_net.state_dict(), os.path.join(self.model_path, f'model_{save_num}.pt'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.optim_path, f'optimizer_{save_num}.pt'))
        self.scheduler.step()
