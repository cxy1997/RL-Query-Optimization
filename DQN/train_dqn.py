from job_env import JOB_env
from dqn_agent import *
import argparse
from dqn_utils import *
import sys
import random
import numpy as np
import torch
import pdb
from tqdm import tqdm, trange
from copy import deepcopy


reward_modes = ["direct_c", "log_c", "log_reduced_c", "log_scale"]


parser = argparse.ArgumentParser()
parser.add_argument('--save-path', type=str, default='run_log_c2')
parser.add_argument('--load-model', type=str, default='off', choices=["on", "off"])
parser.add_argument('--eval', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--buffer-size', type=int, default=50000)
parser.add_argument('--learning-starts', type=int, default=3000)
parser.add_argument('--learning-freq', type=int, default=20)
parser.add_argument('--eval-every', type=int, default=3000)
parser.add_argument('--target-update-freq', type=int, default=25)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epsilon-frames', type=int, default=50000)
parser.add_argument('--hidden-size', type=int, default=32)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--reward-mode', type=str, default='log_c', choices=reward_modes)
parser.add_argument('--agent', type=str, default='dqn', choices=["dqn", "greedy"])


def train(env, dqn_agent, args, num):
    last_obs, _ = env.reset()
    rewards, epi_rewards = 0, []
    eval_target = max(num, args.learning_starts)
    for t in trange(num, 2000000):
        action = dqn_agent.sample_action(last_obs, t)
        obs, reward, done, info = env.step(action)
        rewards += reward[args.reward_mode]
        dqn_agent.push(last_obs, action, obs, reward, done)
        last_obs = obs

        # print('Step %d, Action %d Reward %d' % (t, action, reward), end='\r')

        if done:
            print(f"step {t}, reward {rewards}")
            with open(os.path.join(args.save_path, "train_reward.txt"), 'a') as f:
                f.write(f'Train: Step {t} Reward {rewards}\n')
            if t > eval_target:
                eval_target += args.eval_every
                epi_rewards = test(env, dqn_agent, args, step=t)
                with open(os.path.join(args.save_path, "train_reward.txt"), 'a') as f:
                    f.write(f'Test: Step {t} Reward {epi_rewards}\n')

            rewards = 0
            last_obs, _ = env.reset()

        if t % args.learning_freq == 0 and t > args.learning_starts and dqn_agent.can_sample(args.batch_size):
            dqn_agent.train_model(args.batch_size, t)


def test(env, dqn_agent, args, step=-1):
    print("start testing ...")
    epi_rewards = {x: [] for x in reward_modes}
    for qid in trange(len(env)):
        last_obs, _ = env.reset(qid)
        rewards = {x: 0 for x in reward_modes}
        done = False
        while not done:
            if args.agent == "greedy":
                action = env.get_greedy_action()
            else:
                action = dqn_agent.sample_action(last_obs, args.epsilon_frames+1)
            obs, reward, done, info = env.step(action)
            for k in rewards.keys():
                rewards[k] += reward[k]
            if not args.eval:
                dqn_agent.push(last_obs, action, obs, reward, done)
            last_obs = obs
        print(f'query {qid}, episode reward {rewards}')
        for k in epi_rewards.keys():
            epi_rewards[k].append(rewards[k])

    with open(os.path.join(args.save_path, "test_reward.txt"), 'a') as f:
        f.write(f'Step {step} Reward {str(epi_rewards)}\n')
    for k in epi_rewards.keys():
        epi_rewards[k] = np.mean(epi_rewards[k])
    print('step', step, 'reward', epi_rewards)
    return epi_rewards


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = JOB_env()
    exploration = PiecewiseSchedule([(0, 1.0), (args.epsilon_frames, 0.0)], outside_value=0.0)
    if args.agent == "dqn":
        agent = DQNAgent(args, exploration, args.save_path)
    else:
        agent = None

    if (args.load_model == "on" or args.eval) and agent is not None:
        num = agent.load_model()
    else:
        num = 0

    if not args.eval:
        train(env, agent, args, num)
    else:
        test(env, agent, args)
