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


parser = argparse.ArgumentParser()
parser.add_argument('--save-path', type=str, default='run2')
parser.add_argument('--load-model', type=str, default='on', choices=["on", "off"])
parser.add_argument('--eval', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--buffer-size', type=int, default=100000)
parser.add_argument('--learning-starts', type=int, default=1000)
parser.add_argument('--learning-freq', type=int, default=100)
parser.add_argument('--eval-every', type=int, default=2000)
parser.add_argument('--target-update-freq', type=int, default=6)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epsilon-frames', type=int, default=20000)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=512)


def train(env, dqn_agent, args, num):
    last_obs, _ = env.reset()
    rewards, epi_rewards = 0, []
    eval_target = max(num, args.learning_starts)
    for t in trange(num, 2000000):
        action = dqn_agent.sample_action(last_obs, t)
        obs, reward, done, info = env.step(action)
        rewards += reward
        if not done:
            dqn_agent.push(last_obs, action, obs, reward, done)
        last_obs = obs

        # print('Step %d, Action %d Reward %d' % (t, action, reward), end='\r')

        if done:
            print(f"step {t}, reward {rewards}")
            if t > eval_target:
                eval_target += args.eval_every
                epi_rewards = test(env, dqn_agent, args, step=t)

            rewards = 0
            last_obs, _ = env.reset()

        if t % args.learning_freq == 0 and t > args.learning_starts and dqn_agent.can_sample(args.batch_size):
            dqn_agent.train_model(args.batch_size, t)


def test(env, dqn_agent, args, step=-1):
    print("start testing ...")
    epi_rewards = []
    for qid in trange(len(env)):
        env.reset(qid)
        last_obs, _ = env.reset()
        rewards = 0
        done = False
        while not done:
            action = dqn_agent.sample_action(last_obs, args.epsilon_frames+1)
            obs, reward, done, info = env.step(action)
            rewards += reward
            if not args.eval and not done:
                dqn_agent.push(last_obs, action, obs, reward, done)
            last_obs = obs
        epi_rewards.append(rewards)
        print(f'query {qid}, episode reward {np.mean(epi_rewards):0.4f}')
    print('step', step, 'reward', np.mean(epi_rewards))
    with open(os.path.join(args.save_path, "test_reward.txt"), 'a') as f:
        f.write(f'Step {step} Reward {np.mean(epi_rewards)}\n{str(epi_rewards)}\n')
    return epi_rewards


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = JOB_env()
    exploration = PiecewiseSchedule([(0, 1.0), (args.epsilon_frames, 0.0)], outside_value=0.0)
    dqn_agent = DQNAgent(args, exploration, args.save_path)

    if args.load_model == "on" or args.eval:
        num = dqn_agent.load_model()
    else:
        num = 0

    if not args.eval:
        train(env, dqn_agent, args, num)
    else:
        test(env, dqn_agent, args)
