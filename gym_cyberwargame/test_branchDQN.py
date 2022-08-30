import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
    BranchingDQNPolicy
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, BranchingNet
from supersuit import pad_observations_v0, pad_action_space_v0
from gym_cyberwargame.envs.cyberwargame_env_multiagent import env


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--n-step', type=int, default=2)                # used in DQNPolicy(), estimation_step, number of step to look ahead; less for less loss
    parser.add_argument('--target-update-freq', type=int, default=0)  # the target network update frequency
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=2000)
    # trainer will collect “step_per_collect” transitions and do policy network update repeatedly in each epoch.
    # 实际调用是collector的collect中的n_step
    parser.add_argument('--step-per-collect', type=int, default=200)
    # Used for BranchDQNPolicy()
    parser.add_argument("--action-per-branch", type=int, default=5)
    parser.add_argument("--num-branches", type=int, default=5)
    parser.add_argument("--common-hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128, 128])
    parser.add_argument("--action-hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128, 128])
    parser.add_argument("--value-hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128, 128])

    parser.add_argument('--update-per-step', type=float, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--win-rate',
        type=float,
        default=0.6,
        help='the expected winning rate: Optimal policy can get 0.7'
    )
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--agent-id',
        type=int,
        default=2,
        help='the learned agent plays as the'
        ' agent_id-th player. Choices are 1 and 2.'
    )
    parser.add_argument(
        '--resume-path',
        type=str,
        default='',
        help='the path of agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--opponent-path',
        type=str,
        default='',
        help='the path of opponent agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env():
    cbw_env_pz = env()
    cbw_env_pz = pad_observations_v0(cbw_env_pz)
    cbw_env_pz = pad_action_space_v0(cbw_env_pz)
    cbw_ts = PettingZooEnv(cbw_env_pz)
    return cbw_ts

def get_agents(
    args: argparse.Namespace = get_args(),
    agent_attacker: Optional[BasePolicy] = None,
    agent_defender: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    if_agent_attacker_random = False,
    if_agent_defender_random = False
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    print("get state_shape {}, action_shape {}".format(args.state_shape, args.action_shape))
    args.num_branches = 15

    if agent_attacker is None:
        # model
        """
        a_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device
        ).to(args.device)

        print("Running Device  {} {}".format(args.device, torch.cuda.is_available()))

        if optim is None:
            optim = torch.optim.Adam(a_net.parameters(), lr=args.lr)
        agent_attacker = DQNPolicy(
            a_net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq,
            clip_loss_grad=True
        )
        if args.resume_path:    # no run
            agent_attacker.load_state_dict(torch.load(args.resume_path))
        """
        a_net = BranchingNet(
            state_shape= args.state_shape,
            num_branches=args.num_branches,
            action_per_branch=25, # args.action_per_branch,
            common_hidden_sizes=args.common_hidden_sizes,
            value_hidden_sizes=args.value_hidden_sizes,
            action_hidden_sizes=args.action_hidden_sizes,
            device=args.device,
        ).to(args.device)
        a_optim = torch.optim.Adam(a_net.parameters(), lr=args.lr)
        agent_attacker = BranchingDQNPolicy(
            a_net, a_optim, discount_factor=0.8, target_update_freq=args.target_update_freq, is_double=False
        )
        print("Running Device  {} {}".format(args.device, torch.cuda.is_available()))


    if agent_defender is None:
        if args.opponent_path:
            agent_opponent = deepcopy(agent_attacker)
            agent_opponent.load_state_dict(torch.load(args.opponent_path))
        else:
            agent_opponent = RandomPolicy()

    # model
    d_net = BranchingNet(
        args.state_shape,
        args.num_branches,
        args.action_per_branch,
        args.common_hidden_sizes,
        args.value_hidden_sizes,
        args.action_hidden_sizes,
        device=args.device,
    ).to(args.device)
    d_optim = torch.optim.Adam(d_net.parameters(), lr=args.lr)
    agent_defender = BranchingDQNPolicy(
        d_net, d_optim, args.gamma, target_update_freq=args.target_update_freq
    )
    """
    # Test for Defender as DQNPolicy()
    agent_defender = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq
    )
    """
    # agent_defender = RandomPolicy()
    # 0-attacker, 1-defender
    if if_agent_attacker_random:
        agent_attacker = RandomPolicy()
    elif if_agent_defender_random:
        agent_defender = RandomPolicy()

    agents = [agent_attacker, agent_defender]
    print("Getting Agents {}".format(agents[0]))
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents

# Logger Class to record
class CBW_Logger:

    def __init__(self, writer):
        self.cnt = 0
        self.writer = writer
        self.user_reached_step = []
        self.attacker_reached_step = []
        self.attack_reward_step = []
        self.defend_reward_step = []
        self.util_step = []

        self.user_reached = []
        self.attacker_reached = []
        self.attack_reward = []
        self.defend_reward = []
        self.util = []
        # Track win epochs
        self.a_done = 0
        self.d_done = 0
    def preprocess_fn(self, **kwargs):
        # modify info before adding into the buffer, and recorded into tfb
        # if obs && env_id exist -> reset
        # if obs_next/rew/done/info/env_id exist -> normal step
        if 'rew' in kwargs:
            # print(kwargs)
            info = kwargs['info']
            # print(info)
            # print("!!!!! in preprocess, get info {}".format(info))
            for key in info.keys():
                if key == 'user_reached' or key == 'attacker_reached' or key == 'attack_reward' or key=='util'\
                        or key == 'defend_reward' or key =='a_done' or key=='d_done':
                    self.a_done += int(info.a_done)
                    self.d_done += int(info.d_done)
                    if self.cnt%2==0:
                        self.attack_reward_step.append(info.attack_reward)
                    else:
                        self.attacker_reached_step.append(info.attacker_reached)
                        self.user_reached_step.append(info.user_reached)
                        self.defend_reward_step.append(info.defend_reward)
                        self.util_step.append(info.util)
                    break
            self.cnt += 1
            # print("Self.attacker_reached {} {}".format(self.user_reached,self.attacker_reached ))
            # print("Type of obs now is {}".format(type(obs)))
            # print("In rew, Batch(obs=obs, info=info) {}".format(Batch(obs_next=[obs_next], rew=[rew], info=[info], done=[done])))
            return Batch(info=info)
        else:
            # Call at reset()
            self.cnt = 0
            self.user_reached.append(self.user_reached_step)
            self.attacker_reached.append(self.attacker_reached_step)
            self.attack_reward.append(self.attack_reward_step)
            self.util.append(self.util_step)
            self.defend_reward.append(self.defend_reward_step)

            self.user_reached_step = []
            self.defend_reward_step = []
            self.attacker_reached_step = []
            self.attack_reward_step = []
            self.util_step = []
            # pass
            # print(kwargs)
            return Batch()

    @staticmethod
    def single_preprocess_fn(**kwargs):
        # same as above, without tfb
        if 'rew' in kwargs:
            info = kwargs['info']
            info.rew = kwargs['rew']
            return Batch(info=info)
        else:
            return Batch()

def get_logger():
    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, 'cbw_tianshou_pettingzoo', 'dqn')
    # print("log_path {}".format(log_path))
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)
    cbw_logger = CBW_Logger(writer)

    return logger, cbw_logger


def train_agent(
    args: argparse.Namespace = get_args(),
    agent_attacker: Optional[BasePolicy] = None,
    agent_defender: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    logger=None,
    cbw_logger=None
) -> Tuple[dict, BasePolicy, BasePolicy]:

    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, optim, agents = get_agents(
        args, agent_attacker=agent_attacker, agent_defender=agent_defender, optim=optim,
        if_agent_attacker_random=False, if_agent_defender_random=False
    )

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
        preprocess_fn=cbw_logger.preprocess_fn
    )

    test_collector = Collector(policy, test_envs, exploration_noise=True)
    test_collector = None
    # policy.set_eps(1)
    # train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, 'model_save_path'):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, 'tic_tac_toe', 'dqn', 'policy.pth'
            )
        torch.save(
            policy.policies[agents[args.agent_id - 2]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        # print("Policy {}, and agent id {}".format(policy.policies, args.agent_id))
        policy.policies[agents[args.agent_id-2]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 2]].set_eps(args.eps_test)


    """
    f(rewards: np.ndarray with shape (num_episode, agent_num)) -> np.ndarray
        with shape (num_episode,)
    Use the scalar reward value during training
    """
    def reward_metric(rews):
        # print("Getting rews {}".format(rews))
        return rews[:, args.agent_id - 2]
    # ======== trainer =========
    # Use self-defined train process

    result = offpolicy_trainer(
        policy,                 # Policy contains two policies for attacker and defender
        train_collector,
        test_collector,
        args.epoch,             # Max epoch
        args.step_per_epoch,    # the number of transitions collected per epoch.
        args.step_per_collect,  #  trainer will collect “step_per_collect” transitions and do some policy network update repeatedly in each epoch.
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        # stop_fn=stop_fn,
        # save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy


if __name__ == '__main__':

    args = get_args()
    logger, cbw_logger = get_logger()
    result_multi, policy = train_agent(args, logger=logger, cbw_logger=cbw_logger)

    import matplotlib.pyplot as plt
    import numpy as np
    import datetime
    import os
    from scipy.interpolate import make_interp_spline


    def smooth(scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed


    attacker_reached = cbw_logger.attacker_reached
    user_reached = cbw_logger.user_reached
    attacker_rew = cbw_logger.attack_reward
    util = cbw_logger.util
    defend_reward = cbw_logger.defend_reward
    # print(attacker_rew[1])
    rew_array_all = []
    rew_array_defend = []
    user_reached_all = []
    attacker_reached_all = []
    a_done = cbw_logger.a_done
    d_done = cbw_logger.d_done
    for i in range(len(attacker_rew)):
        if i <= 0:
            continue
        for j in range(len(attacker_rew[i])):
            rew_array_all.append(float(attacker_rew[i][j]))
            rew_array_defend.append(float(defend_reward[i][j]))
    # print(rew_array_defend)
    # 1. print reward graphs
    fig1, ax = plt.subplots(figsize=(12, 8))
    # ax.plot(attacker_reached[::2], attacker_rew[::2], label='attacker')
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    fig6, ax6 = plt.subplots(figsize=(12, 8))
    fig7, ax7 = plt.subplots(figsize=(12, 8))

    # 2. set title and axis
    ax.set_xlabel('Attacker Packets Reached')
    ax.set_ylabel('Attacker Reward')
    ax.set_title('Attacker & Reward ')

    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Reward (Epochs start at 0)')
    ax2.set_title('Attacker Reward Learning Curve')

    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Utilization of the Server')
    ax3.set_title('Server Utilization Monitor')

    ax4.set_xlabel('Epochs Done')
    ax4.set_ylabel('Steps')
    ax4.set_title('Learning Steps')

    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Defender Reward Learning Curve')
    ax5.set_title('Defend Rewards')

    ax6.set_xlabel('Steps')
    ax6.set_ylabel('Attacker VS Defender Reward Learning Curve')
    ax6.set_title('Rewards')

    ax7.set_xlabel('Epochs')
    ax7.set_ylabel('Packets Reached')
    ax7.set_title('Attacker VS User Packets Reached The Server')

    step_take = []
    # 3. Draw the reward for each epochs
    for i in range(len(attacker_rew)):
        # ax.plot(attacker_reached[i], attacker_rew[i], label='attacker')
        # ax2.plot(attacker_rew[i], label='attacker')
        # Util smooth
        """
        if i>=len(attacker_rew)-5:
            util_x = np.array(range(len(util[i])))
            util_y = util[i]
            X_Y_Spline = make_interp_spline(util_x, util_y)
            X_ = np.linspace(util_x.min(), util_x.max(), 100)
            Y_ = X_Y_Spline(X_)
            ax3.plot(util_x, util_y)
        """
        if i >= 1:
            util_x = np.array(range(len(util[i])))
            util_y = np.array(util[i])
            util_y_sm = smooth(util_y, 0.9)
            ax3.plot(util_x, util_y_sm)
            if i % 10 == 0 or i == 1:
                ax3.annotate('Epoch-' + str(i), xy=(util_x[-1], util_y_sm[-1]),
                             xytext=(util_x[-1], util_y_sm[-1]), arrowprops=dict(facecolor='black', shrink=0.05))
            # ax3.plot(util_x, util_y)
            step_take.append(len(attacker_rew[i]))
            # print(len(attacker_reached[i]), attacker_reached[i])
            for item in user_reached[i]:
                user_reached_all.append(float(item))
            for item in attacker_reached[i]:
                attacker_reached_all.append(float(item))
            # user_reached_all.append(np.asarray(user_reached[i], dtype=float))
            # attacker_reached_all.append(np.asarray(attacker_reached[i], dtype=float))

    # ax.plot(attacker_reached, attacker_rew, label='defender')
    # ax.plot(user_reached, defend_reward, label='defender')
    # 2. print utilization graph
    ax2.plot(rew_array_all, linestyle='--', label='attacker')
    # ax2.plot([0]*len(rew_array_all), label='bottom')
    ax3.plot([0.4] * max(len(i) for i in attacker_rew), linestyle='solid', label='threshold')
    ax4.plot(step_take, label='Steps Attacker Take')
    ax5.plot(rew_array_defend, linestyle='-', label='defender')
    ax6.plot(rew_array_defend, linestyle='-', label='defender')
    ax6.plot(rew_array_all, label='attacker')
    ax7.plot(range(len(user_reached_all)), smooth(user_reached_all, 0.9), label='user')
    ax7.plot(range(len(attacker_reached_all)), smooth(attacker_reached_all, 0.9), label='attacker')

    # ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()


    # Save Figure in file with format in Time-
    def save_train():
        train_args = get_args()
        step_per_epoch = train_args.step_per_epoch
        step_per_collect = train_args.step_per_collect
        now_time = datetime.datetime.now()

        script_dir = os.path.dirname('train_results/')
        results_dir = os.path.join(script_dir, now_time.strftime("%m-%d %H-%M") + '-epoch_step-' + str(
            step_per_epoch) + ' collect_step-' + str(step_per_collect) + '/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        fig1.savefig(os.path.join(results_dir, 'attacker_reward_Packets.png'))
        fig2.savefig(os.path.join(results_dir, 'attacker_reward.png'))
        fig3.savefig(os.path.join(results_dir, 'Server Utilization.png'))
        fig4.savefig(os.path.join(results_dir, 'Attacker Learning_step.png'))
        fig5.savefig(os.path.join(results_dir, 'defender_reward.png'))
        fig6.savefig(os.path.join(results_dir, 'Defender VS Attacker.png'))
        fig7.savefig(os.path.join(results_dir, 'Attacker & User Reached.png'))


    print("Attacker wins {}, And defender wins {}".format(a_done, d_done))
    plt.show()
    # save_train()