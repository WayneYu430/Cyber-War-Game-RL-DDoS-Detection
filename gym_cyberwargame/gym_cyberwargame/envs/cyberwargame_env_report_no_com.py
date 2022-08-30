"""
RL environment for a cyber wargame Markov game
"""
from ast import arg
from collections import defaultdict, OrderedDict
import enum
from msilib.schema import Error
import random
from tracemalloc import start
# from random import random
import gym
from gym import spaces
from gym.spaces.utils import flatten_space, flatten, unflatten
import pygame
import numpy as np
from numpy.random import default_rng
# from gym_cyberwargame.envs.host import hostObj
from gym_cyberwargame.network_sim.host import hostObj
from gym.utils import seeding
# SimPy for discrete network events simulation
from gym_cyberwargame.network_sim.attacker import Attacker
from gym_cyberwargame.network_sim.defender import Defender
from gym_cyberwargame.network_sim.user import User
from gym_cyberwargame.network_sim.SimComponents import PacketGenerator, PacketSink, SwitchPort, PortMonitor
import simpy
import random
import functools
# Change to multiagent support environment = PettingZoo
# https://www.pettingzoo.ml/environment_creation#example-custom-environment
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gym_cyberwargame.cbw_logger.cbw_logger import CBW_Logger
import os


def env():
    env = CyberWarGameEnvPettingZoo()
    return env


class CyberWarGameEnvPettingZoo(AECEnv):
    metadata = {"render_modes": ["human"]}
    env_args = {
        "bot_net_num": 5,
        "user_num": 10,
        "utilization": 0.0,
        "interval_num": 5,
        "thresh_utilization": 0.4,
        "upper_bandwidth": 1,
        "reward_lambda": 0.9,
        "reward": 0,
        "time_one_step": 1,  # 1000ms = 1s
        "pkt_attack_reached_num": 0,
        "pkt_noise_reached_num": 0,
        "bytes_noise_reached": 0.0,
        "bytes_attack_reached": 0.0,
        "botnet_src_ip_addr": None,
        "user_src_ip_addr": None,
        "server_ip_addr": 0,
        "units": 1000000000,  # 1,000,000,000 = 1 Gbps for units
        "pkt_size": 64,
        "pkt_per_sec": 10,
        "save_dir": None,
        "initial_state_path": None,
    }

    def __init__(self, env_args=env_args):
        self.pkt_attack_reached_num = env_args['pkt_attack_reached_num']
        self.pkt_noise_reached_num = env_args['pkt_noise_reached_num']
        self.pkt_noise_reached_num = env_args['pkt_noise_reached_num']
        self.bytes_noise_reached = env_args['bytes_noise_reached']
        self.bytes_attack_reached = env_args['bytes_attack_reached']

        # Config for network hosts
        self.server_host = hostObj(0, 0.0, 0.0)  # ip_addr, pkt_sending,pkt_rate
        self.botnet_array = []  # array of hostObj
        self.user_array = []  # array of hostObj
        self.network_array = []  # record of whole network
        self.botnet_src_ip_addr = env_args['botnet_src_ip_addr']
        self.user_src_ip_addr = env_args['user_src_ip_addr']
        self.server_ip_addr = env_args['server_ip_addr']
        self.bot_net_num = env_args['bot_net_num']
        self.user_num = env_args['user_num']

        # Config for network state
        self.utilization = env_args['utilization']
        self.interval_num = env_args['interval_num']  # 10 interval
        self.thresh_utilization = env_args['thresh_utilization']  # [0, 1]
        self.units = env_args['units']  # scale units in 1000000000 = 10^9 = 1Gbps
        self.pkt_per_sec = env_args['pkt_per_sec']  # 100000 pkt/s = 100 Kpkt/s = 6.4Mbps
        self.pkt_size = env_args['pkt_size']  # 64 Bytes = 1 Pkt
        self.time_one_step = env_args['time_one_step']  # 10s
        self.upper_bandwidth = env_args['upper_bandwidth']  # 10000000 bits/s = 1000 Mbps = 10 Gbps

        self.reward_lambda = env_args['reward_lambda']
        self.reward = env_args['reward']

        self.possible_agents = ["attacker", "defender"]  # 0-attacker, 1-defender
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_idx = self.possible_agents
        self.defend_action_branches = self.user_num + self.bot_net_num
        self.simpy_env = simpy.Environment()
        rng = default_rng()
        tmp_ip_addr_array = rng.choice(20, size=1 + self.bot_net_num + self.user_num, replace=False)
        self.server_ip_addr = int(tmp_ip_addr_array[:1])
        self.botnet_src_ip_addr = tmp_ip_addr_array[1:1 + self.bot_net_num]
        self.user_src_ip_addr = tmp_ip_addr_array[1 + self.bot_net_num:]

        self.botnet = Attacker(self.botnet_src_ip_addr, self.time_one_step)
        self.botnet_array = self.botnet.botnet_array
        self.user = User(self.user_src_ip_addr, self.time_one_step)
        self.user_array = self.user.user_array

        self.defender = Defender(self.simpy_env, 10000000).get_defender()
        self.done_with_attacker = 0
        self.done_with_defender = 0
        obs_list_low_a= []
        obs_list_high_a = []
        obs_list_low_d= []
        obs_list_high_d = []

        self._observation_space = {
            "attacker": spaces.Box(np.array(obs_list_low_a), np.array(obs_list_high_a)),
            "defender": spaces.Box(np.array(obs_list_low_d), np.array(obs_list_high_d))
        }

        self.attacker_action_space = spaces.Discrete(self.interval_num * self.bot_net_num, start=0)
        self.attacker_action_map = self._get_attack_action_map()
        self.defender_action_space = spaces.Box(low=0, high=1, shape=(self.bot_net_num + self.user_num,))

        self._action_space = {
            "attacker": self.attacker_action_space,
            "defender": self.defender_action_space
        }

        self.window = None
        self.clock = None
        path = r"."
        path = 'cbw_env_multi.log'
        self.cbw_logger = CBW_Logger(path, level='error', output=False)


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        if agent == "attacker" or agent == 0:
            return self._observation_space["attacker"]
        elif agent == "defender" or agent == 1:
            return self._observation_space["defender"]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent == "attacker" or agent == 0:
            return self._action_space["attacker"]
        elif agent == "defender" or agent == 1:
            return self._action_space["defender"]

    def observe(self, agent):
        tmp_obs = self._get_obs(agent)
        return tmp_obs

    def reset(self, return_info=True, seed=None, options=None):
        self.utilization = 0.0
        self.pkt_reached_num = 0.0
        self.pkt_attack_reached_num = 0  # packets for attacker in eposide
        self.pkt_noise_reached_num = 0  # packets for user in eposide
        self.pkt_attack_num = 0  # packets for attacker in one step
        self.pkt_noise_num = 0  # packets for user in one step
        self.bytes_noise_reached = 0  # bytes for user in one step
        self.bytes_attack_reached = 0  # bytes for attacker in one step
        self.pkt_attack_num_before = 0  # packets for attacker in one step before drop
        self.pkt_noise_num_before = 0  # packets for user in one step before drop
        self.simpy_env = simpy.Environment()
        self.defender = Defender(self.simpy_env, 10000000000).get_defender()  # 10 GBytes

        # Moving to PettingZoo
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.reward = 0
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: {} for agent in self.agents}  # Used for action store
        self.observations = self._get_obs()
        self.step_num_test = 0
        self.keep_util = False
        self.keep_util_count = 0
        self.attack_ack = defaultdict(int)

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        for i in self.botnet_array:
            i.pkt_sending = 0
            i.time_interval = 0

        for i in self.user_array:
            i.pkt_sending = 0

        samp_dist = functools.partial(random.expovariate, 1.0)
        self.user_pg_list = self.user.noise_pg(self.simpy_env)
        for i in self.user_pg_list:
            # print(i)
            i.out = self.defender
        # Use a PM to see the switch
        self.port_monitor = PortMonitor(self.simpy_env, self.defender, samp_dist)
        self.packet_sink = PacketSink(self.simpy_env, debug=False, rec_arrivals=True)

        self.defender.out = self.packet_sink

        observation = self._get_obs()
        info = self._get_info()

        if return_info:
            return observation, info
        else:
            return observation


    def step(self, action):
        agent = self.agent_selection  # 1. "attacker", 2. "defender"
        self._cumulative_rewards[agent] = 0
        self.state[self.agent_selection] = action
        self.cbw_logger.logger.info("Get original action {}".format(action))
        attack_action = None
        defend_action = None
        if agent == "attacker":
            attack_action_ip = random.randint(0, 2)
            attack_action_time = random.randint(3, 5)
            attack_action = attack_action_ip * attack_action_time
            self.cbw_logger.logger.info("Get attack_action {}".format(attack_action))
        elif agent == "defender":
            defend_action = action
            self.cbw_logger.logger.info("Get defend_action {}".format(defend_action))

        if agent == "attacker":
            self.step_num_test += 1
            self.cbw_logger.logger.debug("0. calling attacker with step {}".format(self.step_num_test))
            interval_num, src_ip_addr = self.interpret_attack_action(action=attack_action)
            attack_pg_list = self.attack(interval_num, src_ip_addr)
            for i in attack_pg_list:
                i.out = self.defender

            self.simpy_env.run(until=self.simpy_env.now + self.time_one_step)
            pkt_attack_num = 0
            for i in attack_pg_list:
                pkt_attack_num += i.packets_sent
                for attack_host in self.botnet_array:
                    if i.id == attack_host.ip_addr:
                        attack_host.pkt_sending = i.packets_sent
            self.pkt_attack_num_before = pkt_attack_num

        elif agent == "defender":
            self.step_num_test += 1
            self.cbw_logger.logger.debug("0. calling defender {}".format(self.step_num_test))
            packets_items = self.packet_sink.get_pkt_items()
            pkt_noise_num = self.user_noise()
            self.pkt_noise_num_before = pkt_noise_num
            pkt_attack_num = self.pkt_attack_num_before
            self.cbw_logger.logger.debug(
                "1. Before Dropping: attack sent {}, User sent {}".format(pkt_attack_num, pkt_noise_num))

            pkt_attack_num, pkt_noise_num, \
            bytes_attack_reached_one, bytes_noise_reached_one, \
            pkt_drop_num, attack_ack = self.defend_util(packets_items, pkt_attack_num, pkt_noise_num, defend_action)

            self.cbw_logger.logger.debug("3. Current information for botnet \n{}, \n User Array\n{}".
                                         format(self.botnet.show_attack_pg_list(), self.user.show_user_noise()))
            self.pkt_attack_num = pkt_attack_num
            self.pkt_noise_num = pkt_noise_num
            self.pkt_attack_reached_num += pkt_attack_num  # return to the info, make graph
            self.pkt_noise_reached_num += pkt_noise_num
            self.bytes_attack_reached = bytes_attack_reached_one
            self.bytes_noise_reached = bytes_noise_reached_one
            # 4. Calculate utlization
            cur_util = self.calculate_utilization_one_step(pkt_attack_num, pkt_drop_num, pkt_noise_num
                                                           , self.bytes_attack_reached, self.bytes_noise_reached)
            # Done logic
            if self.utilization >= self.thresh_utilization:  # will done???
                self.keep_util = True
                self.keep_util_count += 1
            else:
                self.keep_util = False
                self.keep_util_count = 0
            if self.utilization >= 1.0:
                self.utilization = 1.0
            if self.keep_util and self.keep_util_count >= 10:
                # self.rewards["attacker"] = 1
                self.dones["defender"] = True
                self.dones["attacker"] = True
                self.rewards["defender"] -= 10
                self.rewards["attacker"] += 10
                self.done_with_attacker += 1
                self.cbw_logger.logger.info(
                    "The game Done with Attacker {}, with utilization keep {}".format(self.rewards,
                                                                                      self.keep_util_count))
                return self._get_obs(), self.rewards, self.dones, self.infos
            if self.step_num_test >= 400:
                self.dones["defender"] = True
                self.dones["attacker"] = True
                self.rewards["defender"] += 10
                self.rewards["attacker"] -= 10
                self.done_with_defender += 1
                self.cbw_logger.logger.info(
                    "The game Done with Defender {}, with utilization keep {}".format(self.rewards,
                                                                                      self.keep_util_count))
                return self._get_obs(), self.rewards, self.dones, self.infos
            # Attack Reward
            total_pkt = self.packet_sink.packets_rec
            prob_attack = self.pkt_attack_num / self.pkt_attack_num_before  # How many attack packets reached the server
            prob_user = self.pkt_noise_num / self.pkt_noise_num_before  # How many user packets reached the server

            if self.utilization <= 0.2:
                self.rewards["attacker"] += 0
            else:
                self.rewards["attacker"] += 0.8 * prob_attack

            # Defender Reward
            if self.utilization >= self.thresh_utilization:
                self.rewards["defender"] -= 0.5
            else:
                self.rewards["defender"] += 1 * (self.reward_lambda) * (1 - prob_attack) + \
                                            -1 * (1 - self.reward_lambda) * (1 - prob_user)

            self.cbw_logger.logger.info("Get Rewards in one step {}, user pass with {:.2f}, attacker pass with {:.2f}". \
                                        format(self.rewards, prob_user, prob_attack))
        if agent == 'attacker':
            observation = self._get_obs(agent='attacker', attack_ack=self.attack_ack)
        else:
            observation = self._get_obs()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
        info = self._get_info()
        self.infos[agent] = info

        return observation, self.rewards, self.dones, self.infos

    def render(self, mode="human"):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    """
    ================================ Private Method ================================
    """
    def _get_obs(self, agent=None, attack_ack=None):
        attack_list = []
        if attack_ack is None:
            for i in self.botnet_array:
                attack_list.append([float(i.ip_addr), float(i.pkt_sending), 0.0])
        else:
            for i in self.botnet_array:
                attack_list.append([float(i.ip_addr), float(i.pkt_sending), float(attack_ack[i.ip_addr])])
        defender_obs = {}
        defender_list = []
        defender_obs['utilization'] = self.utilization
        net_info = {}
        for i in self.botnet_array:
            net_info[str(i.ip_addr)] = [i.ip_addr, i.pkt_sending]
            defender_list.append([float(i.ip_addr), float(i.pkt_sending)])
        for i in self.user_array:
            net_info[str(i.ip_addr)] = [i.ip_addr, i.pkt_sending]
            defender_list.append([float(i.ip_addr), float(i.pkt_sending)])
        defender_obs['net_info'] = net_info

        if agent and agent == 'attacker':
            return np.array(attack_list, dtype=np.float32)
        elif agent and agent == 'defender':
            return np.array(defender_list, dtype=np.float32)
        else:
            return {"attacker": np.array(attack_list, dtype=np.float32),
                    "defender": np.array(defender_list, dtype=np.float32)}

    def _get_info(self):
        info = {}
        bot_net_str = '; \n'.join("attack-" + str(x) for x in self.botnet_array)
        user_str = '; \n'.join("user  -" + str(x) for x in self.user_array)
        server_str = "server-" + str(self.server_host)
        # Info for reward and packet reached
        info['attacker_reached'] = self.pkt_attack_num  # return to the info, make graph
        info['user_reached'] = self.pkt_noise_num  # return to the info, make graph
        info['attack_reward'] = self.rewards['attacker']
        info['defend_reward'] = self.rewards['defender']
        info['util'] = self.utilization
        info['a_done'] = self.done_with_attacker
        info['d_done'] = self.done_with_defender
        return info

    def __get_attack_obs(self):
        obs_dict = defaultdict(int)
        obs_list_low = []
        obs_list_high = []
        for i in self.botnet_array:
            obs_dict[i.ip_addr] = spaces.Box(low=np.array([0, 0, 0]), high=np.array([255, np.inf, np.inf]), shape=(3,),
                                             dtype=int)
            obs_list_low.append(np.array([0, 0, 0]))
            obs_list_high.append(np.array([255, np.inf, np.inf]))
        __attack_space = spaces.Box(np.array(obs_list_low), np.array(obs_list_high))
        return __attack_space

    def __get_defend_obs(self):
        obs_dict = {}
        obs_dict['utilization'] = spaces.Box(0.0, 1.0, shape=(1,))
        net_info = {}
        obs_list_low = []
        obs_list_high = []
        for i in self.botnet_array:
            net_info[i.ip_addr] = spaces.Box(low=np.array([0, 0]), high=np.array([255, np.inf]), \
                                             shape=(2,), dtype=int)
            obs_list_low.append(np.array([0, 0]))
            obs_list_high.append(np.array([255, np.inf]))
        for i in self.user_array:
            net_info[i.ip_addr] = spaces.Box(low=np.array([0, 0]), high=np.array([255, np.inf]), \
                                             shape=(2,), dtype=int)
            obs_list_low.append(np.array([0, 0]))
            obs_list_high.append(np.array([255, np.inf]))
        obs_dict['net_info'] = spaces.Dict(net_info)

        _defend_space = spaces.Dict(obs_dict)
        _defend_space_box = spaces.Box(np.array(obs_list_low), np.array(obs_list_high))
        return _defend_space_box

    """
    ================================ Helper Functions ================================
    """

    def interpret_attack_action(self, action):
        src_ip_addr, interval_num = self.attacker_action_map[int(action)][0], \
                                    self.attacker_action_map[int(action)][1]
        assert interval_num != 0, ("Interval Num can not be 0")
        return interval_num, src_ip_addr

    def interpret_defend_action(self, action):
        pass

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def max_host_rate(self, ):
        total_host = self.bot_net_num + self.user_num
        max_rate_each = round(self.upper_bandwidth / total_host)
        return max_rate_each


    def attack(self, chosen_interval_num, src_ip_addr, size_dist=0.1):

        attack_pg_list = self.botnet.attack_pg_list(self.simpy_env, src_ip_addr, chosen_interval_num, size_dist)
        return attack_pg_list
        # pass

    def _get_attack_action_map(self, ):
        action_map = defaultdict()
        action_index = self.bot_net_num * self.interval_num
        count = 0
        for bot_i, host in enumerate(self.botnet_array):
            for int_i, int_v in enumerate(range(1, self.interval_num + 1)):  # start from 1
                action_map[count] = [host.ip_addr, int_v]
                count += 1
        assert count == action_index, ("Action index should be same as dict keys()")
        return action_map

    def _get_defend_action_map(self, ):
        action_map = defaultdict()
        action_index = self.bot_net_num * self.user_num
        count = 0
        for bot_i, host in enumerate(self.botnet_array):
            action_map[count] = host.ip_addr
            count += 1
        for host_i, host in enumerate(self.user_array):
            action_map[count] = host.ip_addr
            count += 1

        assert count == action_index, ("Action index should be same as dict keys()")
        return action_map

    def defend_util(self, packets_items, pkt_attack_num, pkt_noise_num, action):
        bytes_attack_reached_one = 0.0
        bytes_noise_reached_one = 0.0
        pkt_attack_pass = 0
        pkt_noise_pass = 0
        attack_ack = defaultdict(int)  # How many packets passed by defender, which equals to ACK to attacker
        drop_dict = OrderedDict()  # How many packets need to be dropped
        defend_dict = defaultdict(list)  # A dict for all in-defend records

        defend_drop_volume = [0.0, 0.1, 0.2, 0.3, 0.4]
        for i in self.botnet_array:
            drop_dict[i.ip_addr] = 0
            defend_dict[i.ip_addr] = [0.0, 0]  # Record of [bytes, pkt_num]
        for i in self.user_array:
            drop_dict[i.ip_addr] = 0
            defend_dict[i.ip_addr] = [0.0, 0]

        index = 0
        for k, v in drop_dict.items():
            drop_dict[k] = defend_drop_volume[action[index]]
            index += 1

        packet_array = np.array(packets_items)
        for pkt in packet_array:
            defend_dict[pkt[0]][0] += pkt[1]
            defend_dict[pkt[0]][1] += 1

        pkt_drop_num = 0
        for drop_k, v in drop_dict.items():
            host_bytes_pass = defend_dict[drop_k][0] * (1 - v)
            host_pkt_pass = defend_dict[drop_k][1] * (1 - v)
            pkt_drop_num += int(defend_dict[drop_k][1] * (v))

            if drop_k in self.user_src_ip_addr:
                bytes_noise_reached_one += host_bytes_pass
                pkt_noise_pass += int(host_pkt_pass)
            elif drop_k in self.botnet_src_ip_addr:
                bytes_attack_reached_one += host_bytes_pass
                pkt_attack_pass += int(host_pkt_pass)
                attack_ack[drop_k] = pkt_attack_pass
        pkt_attack_num = pkt_attack_pass
        pkt_noise_num = pkt_noise_pass
        return pkt_attack_num, pkt_noise_num, \
               bytes_attack_reached_one, bytes_noise_reached_one, \
               pkt_drop_num, attack_ack

    def user_noise(self):
        pkt_noise_num = 0
        for i in self.user_pg_list:
            # print("Getting User Noise")
            pkt_noise_num += i.packets_sent
            self.user.update_info(i.id, i.packets_sent)
            i.packets_sent = 0

        return pkt_noise_num

    def calculate_utilization_one_step(self, pkt_attack_num, pkt_drop_num, pkt_noise_num, bytes_attack_reached_one,
                                       bytes_noise_reached_one):

        cur_util = ((bytes_attack_reached_one + bytes_noise_reached_one) * 8) \
                   / (self.upper_bandwidth * self.units * self.time_one_step)
        self.utilization = cur_util
        return self.utilization