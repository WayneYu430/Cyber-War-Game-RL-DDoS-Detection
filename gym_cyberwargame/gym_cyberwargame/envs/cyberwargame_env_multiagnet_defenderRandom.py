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
from matplotlib import units
from matplotlib.style import use
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


def b_random_env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = CyberWarGameEnvPettingZoo()
    # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


class CyberWarGameEnvPettingZoo(AECEnv):
    metadata = {"render_modes": ["human"]}
    env_args = {
        "pkt_attack_reached_num": 0,
        "pkt_noise_reached_num": 0,
        "bytes_noise_reached": 0.0,
        "bytes_attack_reached": 0.0,
        "botnet_src_ip_addr": None,
        "user_src_ip_addr": None,
        "server_ip_addr": 0,
        "bot_net_num": 5,
        "user_num": 5,
        "utilization": 0.0,
        "interval_num": 5,
        "thresh_utilization": 0.4,
        "units": 1000000000,  # 1,000,000,000 = 1 Gbps for units
        "pkt_size": 64,
        "pkt_per_sec": 10,
        "upper_bandwidth": 1,
        "reward_lambda": 0.8,
        "reward": 0,
        "time_one_step": 1,  # 1000ms = 1s
        "save_dir": None,
        "initial_state_path": None,
    }

    def __init__(self, env_args=env_args):
        """
        Initializes the environment

        Observation:
            - Provide the sevrer info: {pkt_reached_num, bot_net_src_ip_addr, server_ip_addr, utilization, service?}
            Type: Discrete(4)
        Actions:
            - Attacker Action: Discrete(number of intervals * number of bot_net_src_ip_addr from a botnet)
                - interval_num
                - bot_net_src_ip_addr
            - Defense Action: Discrete(Drop/ Not Drop the pkt of the server)
        Reward:
            Reward ??? Need symmetric
            - Send to the server pkt
        Starting State:
            - Background user noise: a normal pkt sending rate
            - Attack values is 0
        Episode Termination:
            - When server utilization reach 95%
            - When attacker is detected???


        :param thresh_utilization: set the upper bound of the utilization, indicating the done
        :param pkt_per_sec: the assumption of the pkt could be sent per sec, the interval of time units in second.
        :param time_one_step: time of one step
        :param upper_bandwidth: the bandwidth of the network, units in M pkt per second.

        :param : configuration of the environment
        :param save_dir: directory to save outputs, e.g. initial state
        :param initial_state_path: path to the initial state (if none, use default)
        """
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

        # Use for reward function
        self.reward_lambda = env_args['reward_lambda']
        self.reward = env_args['reward']

        # Use for two agent in Tianshou & Fit to PettingZoo, 0706
        self.possible_agents = ["attacker", "defender"]  # 0-attacker, 1-defender
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_idx = self.possible_agents
        # Used for BranchDQNPolicy For defender to multipule output
        self.defend_action_branches = self.user_num + self.bot_net_num
        # Observations are dictionaries with the server's info
        # Different observation for Attacker and Defender
        """
        Setting the network, only once
        """
        # New Attack&Defend for simulation
        self.simpy_env = simpy.Environment()

        # tmp_ip_addr_array  = self.np_random.integers(1, 255, size=1+self.bot_net_num+self.user_num)
        # tmp_ip_addr_array = np.random.randint(1, 255, size=1+self.bot_net_num+self.user_num)
        rng = default_rng()
        tmp_ip_addr_array = rng.choice(20, size=1 + self.bot_net_num + self.user_num, replace=False)
        self.server_ip_addr = int(tmp_ip_addr_array[:1])
        self.botnet_src_ip_addr = tmp_ip_addr_array[1:1 + self.bot_net_num]
        self.user_src_ip_addr = tmp_ip_addr_array[1 + self.bot_net_num:]

        self.botnet = Attacker(self.botnet_src_ip_addr, self.time_one_step)
        self.botnet_array = self.botnet.botnet_array
        self.user = User(self.user_src_ip_addr, self.time_one_step)
        self.user_array = self.user.user_array

        # Use constant for now 0701
        self.defender = Defender(self.simpy_env, 10000000).get_defender()
        # Track the game done winner
        self.done_with_attacker = 0
        self.done_with_defender = 0
        """
        Observation for Attacker:  {src_ip, pkt_sent, ACK (the pkt reached the server)}
            - Attack should only see the packets of botnet?
            - 1. If attack see the utilization (x)
            - 2. If attack only see the botnet information in a list:  {src_ip, pkt_sent, ACK(the pkt reached the server)} of botnet
        Observation for Defender: utilization, {src_ip_addr, pkt_sent} (array)
            - Defender see the server
            - Can defender see user??? - NO
        """
        self._observation_space = {
            "attacker": self.__get_attack_obs(),
            # "attacker": spaces.Box(np.array([0, 0.0, 0]), np.array([np.inf, 1.0, 255]), ),
            # "defender": spaces.Box(np.array([0.0, 0]), np.array([1.0, 255]), shape=(2,), )
            "defender": self.__get_defend_obs()
        }

        # For attacker, we have different numbers and differetn ip_scr number of attack action. thus. a MultiDiscrete
        # For defender, we have 2 action, {drop/ Not drop} the packet
        """

        self.attacker_action_space = spaces.Dict({
            "interval_num": spaces.Discrete(self.interval_num, start=1),
            "src_ip": spaces.Discrete(self.bot_net_num)
        })  # (time, ip)
        """
        self.attacker_action_space = spaces.Discrete(self.interval_num * self.bot_net_num, start=0)
        """0809 - Change to pure Attacker
        host_num = self.bot_net_num + self.user_num
        a_action = self.bot_net_num * self.interval_num
        self.attacker_action_space = spaces.Box(low=np.array([0] * host_num),
                                                high=np.array([a_action] + [0] * (host_num - 1)), \
                                                shape=(self.bot_net_num + self.user_num,), dtype=np.int32)
        """
        self.attacker_action_map = self._get_attack_action_map()
        # Use for RLlib (X)
        # Defender action space is using a map to drop the packet from an IP
        # self.defender_action_space = spaces.Box(low=0, high=1, shape=(self.bot_net_num + self.user_num,), dtype=np.int32)  # 0 drop and 1 not drop
        self.defender_action_space = spaces.Discrete(self.user_num + self.bot_net_num, start=0)
        # 0802- Change to drop the volume of the packets
        # Action in [.0, 0.1, 0.2, 0.3, 0.4] = Discrete(5)
        # self.defender_action_space = spaces.Discrete(5, start=0)

        # self.defender_action_map = self._get_defend_action_map()
        self._action_space = {
            "attacker": self.attacker_action_space,
            "defender": self.defender_action_space
        }

        # Print environment information at the __init__
        # print(self._get_info())
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        """
        2022-07-14
        - Adding logger feature
        - store necessary steps in file
        """
        # path = 'D:\Wayne_OMEN_Onedrive\OneDrive\Telecommunications\MSc Project\code\gym_cyberwargame\gym_cyberwargame'
        # os.path.join(path, '/cbw_run/', 'cbw_env_multi.log')
        path = r"D:\Wayne_OMEN_Onedrive\OneDrive\Telecommunications' \
               '\MSc Project\code\gym_cyberwargame\gym_cyberwargame\cbw_run\cbw_env_multi.log"
        path = 'cbw_env_multi.log'
        self.cbw_logger = CBW_Logger(path, level='info', output=False)

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
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
            # Flatten the action to NN in/output
            # _action_space_flatten = flatten_space(self._action_space["attacker"])
            #
            return self._action_space["attacker"]
        elif agent == "defender" or agent == 1:
            # _action_space_flatten = flatten_space(self._action_space["defender"])
            return self._action_space["defender"]

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        # Using the Flatten Wrapper in Gym
        tmp_obs = self._get_obs(agent)
        # info = self._get_info()
        # tmp_space = self.observation_space(agent)
        # tmp_obs = flatten(tmp_space, tmp_obs)
        return tmp_obs

    def reset(self, return_info=True, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
            - agents
            - rewards
            - _cumulative_rewards
            - dones
            - infos
            - agent_selection
            And must set up the environment so that render(), step(), and observe()
            can be called without issues.

        Here it sets up the state dictionary which is used by step() and \
        the observations dictionary which is used by step() and observe()
        """
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
        # self.observations = {agent: {} for agent in self.agents}
        self.observations = self._get_obs()
        self.step_num_test = 0
        # Used for keep the utilization for several steps. eg. 5 steps
        self.keep_util = False
        self.keep_util_count = 0
        self.attack_ack = defaultdict(int)
        # Track the game done winner
        # self.done_with_attacker = 0
        # self.done_with_defender = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        """
        - Do not change the network?
        - Do not change the IP in one game!!! Fixed Network Topo => Learn the Botnet
        """
        for i in self.botnet_array:
            i.pkt_sending = 0
            i.time_interval = 0

        for i in self.user_array:
            i.pkt_sending = 0

        """
        Generating the user noise
        Wired the network
        """
        samp_dist = functools.partial(random.expovariate, 1.0)

        self.user_pg_list = self.user.noise_pg(self.simpy_env)
        for i in self.user_pg_list:
            # print(i)
            i.out = self.defender
        # Use a PM to see the switch
        self.port_monitor = PortMonitor(self.simpy_env, self.defender, samp_dist)

        # Use a PS to collect the defender
        self.packet_sink = PacketSink(self.simpy_env, debug=False, rec_arrivals=True)

        self.defender.out = self.packet_sink
        """
        Attack action & defend action
        """
        # Run one time_step
        # self.simpy_env.run(until=self.simpy_env.now+self.time_one_step)

        observation = self._get_obs()
        info = self._get_info()

        # return observation
        self.cbw_logger.logger.info(
            "00000 Calling Reset(), ready to play game")
        if return_info:
            return observation, info
        else:
            return observation

    """
    Chose an action (the index from action_space)
        :params action: a dict of agent_id & agent_action index 
        return (state, reward, done, info)
    """

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        # PettingZoo, Choose Agent
        # if self.dones[self.agent_selection]:
        # action = None
        # self._was_done_step(action)
        #     self.dones['attacker'] = True
        #     self.dones['defender'] = True
        #     return

        agent = self.agent_selection  # 1. "attacker", 2. "defender"
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0
        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act, i.e. First Attack, then, Defender
        # if self._agent_selector.is_last():
        self.cbw_logger.logger.info("Get original action {}".format(action))
        # 0. Assign Action, DQN returns a single scalar value -> int
        attack_action = None
        defend_action = None
        if agent == "attacker":
            attack_action = action
            # print("Get attack_action_value", type(attack_action))
            self.cbw_logger.logger.info("Get attack_action {}".format(attack_action))
        elif agent == "defender":
            defend_action = action
            defend_action = np.random.randint(0, 3, 10)
            self.cbw_logger.logger.info("Get defend_action {}".format(defend_action))

        if agent == "attacker":
            # print("In step ----calling attacker")
            self.step_num_test += 1
            self.cbw_logger.logger.debug("0. calling attacker with step {}".format(self.step_num_test))
            # 1. Attack ation logic, action shape = (time, ip), time can not be 0!
            interval_num, src_ip_addr = self.interpret_attack_action(action=attack_action)
            # pkt_attack_num = self.attack(interval_num, src_ip_addr)
            attack_pg_list = self.attack(interval_num, src_ip_addr)
            for i in attack_pg_list:
                # print("--Attack List of",str(i))
                i.out = self.defender

            # 2. Move a step of simpy_env, get user noist and attack packets
            self.simpy_env.run(until=self.simpy_env.now + self.time_one_step)
            pkt_attack_num = 0
            for i in attack_pg_list:
                pkt_attack_num += i.packets_sent
                # Updating the botnet info, for obs construction
                for attack_host in self.botnet_array:
                    if i.id == attack_host.ip_addr:
                        attack_host.pkt_sending = i.packets_sent
            self.pkt_attack_num_before = pkt_attack_num

        elif agent == "defender":
            # print("In step----calling defender")
            self.step_num_test += 1
            self.cbw_logger.logger.debug("0. calling defender {}".format(self.step_num_test))
            # 3. Defend Logic, simple random choose a src to drop
            # Drop using items list return from the FIFO queue to iterate act drop or not drop
            # - Collect information of packets and bytes.
            # - Construct the observation for attack and defender
            packets_items = self.packet_sink.get_pkt_items()
            # print("length of packets_items is {}".format(len(packets_items)))
            # self.cbw_logger.logger.debug("length of packets_items is {} with {}".format(len(packets_items), packets_items))
            pkt_noise_num = self.user_noise()
            self.pkt_noise_num_before = pkt_noise_num

            pkt_attack_num = self.pkt_attack_num_before
            # print("1. Before Dropping: attack sent {}, User sent {}".format(pkt_attack_num, pkt_noise_num))
            self.cbw_logger.logger.debug(
                "1. Before Dropping: attack sent {}, User sent {}".format(pkt_attack_num, pkt_noise_num))

            # A random defender policy
            pkt_attack_num, pkt_noise_num, \
            bytes_attack_reached_one, bytes_noise_reached_one, \
            pkt_drop_num, attack_ack = self.defend_util(packets_items, pkt_attack_num, pkt_noise_num, defend_action)

            # print("2. After Dropping:  attack reached {}, user reached {}, total dropped: {}".format(pkt_attack_num, pkt_noise_num, pkt_drop_num))
            self.cbw_logger.logger.debug(
                "2. After Dropping:  attack reached {}, user reached {}, total dropped: {}". \
                    format(pkt_attack_num, pkt_noise_num, pkt_drop_num))

            # Print the host_array information for debug the process
            self.cbw_logger.logger.debug("3. Current information for botnet \n{}, \n User Array\n{}".
                                         format(self.botnet.show_attack_pg_list(), self.user.show_user_noise()))
            # Collect packet information
            self.pkt_attack_num = pkt_attack_num
            self.pkt_noise_num = pkt_noise_num
            self.pkt_attack_reached_num += pkt_attack_num  # return to the info, make graph
            self.pkt_noise_reached_num += pkt_noise_num
            self.bytes_attack_reached = bytes_attack_reached_one
            self.bytes_noise_reached = bytes_noise_reached_one

            # Updating the hostObj array of user, botnet
            # Updating attacker
            # print("Attacker ACK Reaching\n",attack_ack)    # Number of packets reached equal to ACK reply, as observation
            self.cbw_logger.logger.debug(
                "Attacker ACK Reaching {}".format(attack_ack))
            # 4. Calculate utlization
            cur_util = self.calculate_utilization_one_step(pkt_attack_num, pkt_drop_num, pkt_noise_num
                                                           , self.bytes_attack_reached, self.bytes_noise_reached)
            self.cbw_logger.logger.info(
                "Current bytes for attack=%s, user=%s, Step cur_util=%s" % \
                (bytes_attack_reached_one, bytes_noise_reached_one, str(self.utilization)))

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

            # if self.step_num_test >= 400 and self.keep_util_count == 0:
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
            if self.pkt_attack_num_before < pkt_attack_num or prob_attack > 1.0:
                assert AssertionError("Cannot be large than before drop, with before{}, now{}".format(self.pkt_attack_num_before, pkt_attack_num))
            # Maybe the attacker can not see the prob_user
            # Use delayed reward for attacker
            if self.utilization <= 0.2:
                self.rewards["attacker"] += 0
            else:
                self.rewards["attacker"] += 1 * prob_attack

            """
            self.rewards["attacker"] += self.reward_lambda * prob_attack - \
                                       (1 - self.reward_lambda) * prob_user
            """
            """
            self.rewards["attacker"] += (1*self.pkt_attack_num - (self.pkt_attack_num_before-self.pkt_attack_num_before)) \
                                            / self.pkt_attack_num_before
            """
            self.reward = self.rewards["attacker"]

            # Defender Reward
            """
            legal_pass = 1 * self.pkt_noise_num
            legal_drop = -1 * (self.pkt_noise_num_before - self.pkt_noise_num)
            attack_pass = -1 * self.pkt_attack_num
            attack_drop = 1 * (self.pkt_attack_num_before - self.pkt_attack_num)
            total_reward = legal_pass + legal_drop + attack_pass + attack_drop
            # self.rewards["defender"] += total_reward
            """
            self.rewards["defender"] += 1 * (self.reward_lambda) * (1 - prob_attack) + \
                                        -1 * (1 - self.reward_lambda) * (1 - prob_user)

            self.cbw_logger.logger.info("Get Rewards in one step {}, user pass with {:.2f}, attacker pass with {:.2f}". \
                                        format(self.rewards, prob_user, prob_attack))
        if agent == 'attacker':
            observation = self._get_obs(agent='attacker', attack_ack=self.attack_ack)
        else:
            observation = self._get_obs()
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        info = self._get_info(agent=agent)
        self.infos[agent] = info
        # return observation, self.rewards["attacker"], self.dones["attacker"], info
        return observation, self.rewards, self.dones, self.infos

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if len(self.agents) == 2:
            string = "Current state: Attacker: {} , Defender: {}".format("Attacker", "Defender")
        else:
            string = "Game over"
        print("================================================================\n")
        print(string)
        print("================================================================\n")

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    """
    ================================ Private Method ================================
    """

    """
    _get_obs(self, agent=None, attack_ack=None):
    - Prepare the state input for RL
    - Change to Numpy to handle the data
    """

    def _get_obs(self, agent=None, attack_ack=None):
        # For attack
        attack_obs = defaultdict(int)
        attack_list = []
        if attack_ack is None:
            for i in self.botnet_array:
                # attack_obs[str(i.ip_addr)] = [i.ip_addr, i.pkt_sending,0]  # {src_ip, pkt_sent, ACK (the pkt reached the server)}
                attack_list.append([float(i.ip_addr), float(i.pkt_sending), 0.0])
        else:
            for i in self.botnet_array:
                # attack_obs[str(i.ip_addr)] = [i.ip_addr, i.pkt_sending, attack_ack[i.ip_addr]]  # {src_ip, pkt_sent, ACK (the pkt reached the server)}
                attack_list.append([float(i.ip_addr), float(i.pkt_sending), float(attack_ack[i.ip_addr])])

        # For defender
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

        # return {"attacker": attack_list, "defender": defender_obs}
        if agent and agent == 'attacker':
            return np.array(attack_list, dtype=np.float32)
        elif agent and agent == 'defender':
            return np.array(defender_list, dtype=np.float32)
        else:
            return {"attacker": np.array(attack_list, dtype=np.float32),
                    "defender": np.array(defender_list, dtype=np.float32)}
        # return attack_list
        # print("received: {}, attack sent {}, User sent {}".format(self.defender.packets_rec, pkt_attack_num, tmp_user_pkt))

    """
    Get the environment info at the begining

    """

    def _get_info(self, agent=None):
        info = {}
        bot_net_str = '; \n'.join("attack-" + str(x) for x in self.botnet_array)
        user_str = '; \n'.join("user  -" + str(x) for x in self.user_array)
        server_str = "server-" + str(self.server_host)
        # Info for reward and packet reached
        info['attacker_reached'] = self.pkt_attack_num  # return to the info, make graph
        info['user_reached'] = self.pkt_noise_num  # return to the info, make graph
        if agent == 'defender':
            if self.pkt_attack_num_before != 0 and self.pkt_noise_num_before != 0:
                info['attacker_reached_p'] = self.pkt_attack_num / self.pkt_attack_num_before
                info['user_reached_p'] = self.pkt_noise_num / self.pkt_noise_num_before
            else:
                info['attacker_reached_p'] = 0.0
                info['user_reached_p'] = 0.0
        else:
            info['attacker_reached_p'] = 0.0
            info['user_reached_p'] = 0.0

        self.cbw_logger.logger.info(
            "Get info with agent{}. attack_P {}, user_P {}".format(agent,info['attacker_reached_p'],
                                                                   info['user_reached_p']))
        info['attack_reward'] = self.rewards['attacker']
        info['defend_reward'] = self.rewards['defender']
        info['util'] = self.utilization
        info['a_done'] = self.done_with_attacker
        info['d_done'] = self.done_with_defender
        # return bot_net_str+'\n'+user_str+'\n'+server_str
        info['agent'] = agent
        return info

    """
    attack_obs={src_ip, pkt_sent, ACK (the pkt reached the server)}

    """

    def __get_attack_obs(self):
        obs_dict = defaultdict(int)
        obs_list_low = []
        obs_list_high = []
        for i in self.botnet_array:
            obs_dict[i.ip_addr] = spaces.Box(low=np.array([0, 0, 0]), high=np.array([255, np.inf, np.inf]), shape=(3,),
                                             dtype=int)
            obs_list_low.append(np.array([0, 0, 0]))
            obs_list_high.append(np.array([255, np.inf, np.inf]))
            # obs_list.append(spaces.Box(low=np.array([0.0, 0]), high=np.array([0.0, 255]), shape=(2,), dtype=np.float64))
            # Regard IP as feature??? Box in 2D
            # obs_dict[str(i.ip_addr)] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
        # __attack_space = spaces.Dict(obs_dict)
        __attack_space = spaces.Box(np.array(obs_list_low), np.array(obs_list_high))
        # print("__attack_space{}".format(obs_dict.items()))
        # __attack_space = spaces.Tuple(obs_list)
        return __attack_space

    """
    __get_defend_obs={utilization: utilization, net_info: {src_ip: {src_ip, pkt_sent}}}

    """

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
    """
    __get_nework_obs__
        :param: network_array: a hostObj list, could be botnet, user_net, server?
        :return: the obs of the sub-network
    """

    def __get_nework_obs__(self, network_array):

        __network_obs_ = np.ndarray([[item.ip_addr, item.pkt_rate] for item in network_array])
        return __network_obs_

    """
    The attack action: Array in (time, ip)
        - Choose a time interval 0
        - Choose a ip_src_addr from the botnet
        :action: [Interval, ip_src]
    """

    def interpret_attack_action(self, action):
        # assert action is list
        # print(action)
        # if not isinstance(action, np.ndarray):
        #     print(type(action))
        #     raise ValueError("action is not ndarray")
        # _unflatten_action = unflatten(self._action_space['attacker'], action)
        # print("Type of action {}, action is {}".format(action.shape, action))

        src_ip_addr, interval_num = self.attacker_action_map[int(action)][0], \
                                    self.attacker_action_map[int(action)][1]
        assert interval_num != 0, ("Interval Num can not be 0")
        # print("Transfrom action {}, interval_num {}, ip_src{}".format(action, interval_num, src_ip_addr) )
        """

        if action is list or action is np.ndarray:
            interval_num = action[0]
            src_ip_addr = self.botnet_array[action[1]].ip_addr
        else:
            # Trying to find in the botnet 
            for i in self.botnet_array:
                for j in range(self.interval_num):
                    if i.ip_addr * j == action:
                        interval_num = j
                        src_ip_addr = i.ip_addr
        """
        return interval_num, src_ip_addr
        # pass

    def interpret_defend_action(self, action):
        pass

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    """
    max_host_rate
    - According the network environment, calculate the maximum data rate
    - e.g. Server 100Mbps, 10 total host (5 for botnet, 5 for user), each host max = 10Mbps
    - In reality: depends on router ports rate & router handling; Queueing theory; delay; etc... 
    - Problem: Can not reach the upper bandwidth???
    """

    def max_host_rate(self, ):
        total_host = self.bot_net_num + self.user_num
        max_rate_each = round(self.upper_bandwidth / total_host)
        return max_rate_each

    """
    The attack: 
    Calculate the number of pkt in one epsiode that attacker send to the server.
        - Choose a time interval 
        - Choose a ip_src_addr from the botnet

        :param chosen_interval_num: the chosen interval number 
        :param src_ip_addr: the ip_src_addr
        :param size_dist: for packetgenerator size distribution
        :return: the number of packet transmitted in this epsiode
    """

    def attack(self, chosen_interval_num, src_ip_addr, size_dist=0.1):

        attack_pg_list = self.botnet.attack_pg_list(self.simpy_env, src_ip_addr, chosen_interval_num, size_dist)
        return attack_pg_list
        # pass

    """
    _get_attack_action_map:
        :return: the attack action map, key is the Discrete action spapce from (botnet_num*intervel_num)
        value is the [scr_ip_addr, interval_num]
    """

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

    """
    _get_defend_action_map
    - In one step, only choose one ip address packet to drop or not drop
    action map like one-hot coding: [0, 0, 0, 0, 0, 0, 1]. Means chose last one ip_src packet to drop
    """

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

    """
    The defender action: {drop / not drop}
    - Random choose a pkt-sending value, and drop to zero.
        :params 
        :return: the number of packet dropped in this step
        :return drop_attack - Indicting dropping packets from attack
    """

    def defend_old(self, action=0, src_ip_addr=0, pkt_drop_prob=1):
        pkt_drop_prob = random.random()
        user_or_botnet = random.randint(0, 1)
        drop_pkt = 0
        drop_attack = False
        if user_or_botnet == 0:
            drop_attack = False
            random_ip = random.randint(0, len(self.user_array) - 1)
            src_ip_addr = self.user_array[random_ip].ip_addr
            for i in self.user_array:
                # print("!!!!!!Updating self.user_array[i] ", i)
                if i.ip_addr == src_ip_addr:
                    drop_pkt = round(i.pkt_sending * pkt_drop_prob)
                    i.pkt_sending = i.pkt_sending - drop_pkt
        elif user_or_botnet == 1:
            drop_attack = True
            random_ip = random.randint(0, len(self.botnet_array) - 1)
            src_ip_addr = self.botnet_array[random_ip].ip_addr
            for i in self.botnet_array:
                # print("!!!!!!Updating self.botnet_array[i] ", i)
                if i.ip_addr == src_ip_addr:
                    drop_pkt = round(i.pkt_sending * pkt_drop_prob)
                    i.pkt_sending = i.pkt_sending - drop_pkt
        return drop_pkt, drop_attack

        """
        The defender action: {drop / not drop}
        - Random choose a pkt-sending value, and drop to zero.
            :params 
            :return: the number of packet dropped in this step
            :return drop_attack - Indicting dropping packets from attack
            :return 
        """

    def defend(self, packets_items, pkt_attack_num, pkt_noise_num, action=None):
        # print("Getting packets_items {}".format(packets_items))
        pkt_drop_num = 0
        bytes_attack_reached_one = 0.0
        bytes_noise_reached_one = 0.0
        attack_ack = defaultdict(int)  # How many packets passed by defender, which equals to ACK to attacker
        # Assume the defender output is (host_num, )
        host_drop_flag = np.random.choice([0, 1], (self.bot_net_num + self.user_num), p=[0.7, 0.3])
        if action is not None:
            host_drop_flag = action
        drop_dict = defaultdict(int)
        count = 0
        for bot in self.botnet_array:
            drop_dict[bot.ip_addr] = host_drop_flag[count]
            count += 1
        for user in self.user_array:
            drop_dict[user.ip_addr] = host_drop_flag[count]
            count += 1
        for i in packets_items:  # src_ip, bytes, flow_id
            if pkt_attack_num <= 0 or pkt_noise_num <= 0:
                break
            # drop_flag = random.randint(0,1)
            drop_flag = drop_dict[i[0]]
            if drop_flag == 1:
                pkt_drop_num += 1
                if i[2] == "attacker":
                    pkt_attack_num -= 1
                elif i[2] == "user":
                    pkt_noise_num -= 1
            elif drop_flag == 0:  # pass the defender

                if i[2] == "attacker":
                    attack_ack[i[0]] += 1
                    bytes_attack_reached_one += float(i[1])
                else:
                    bytes_noise_reached_one += float(i[1])
        return pkt_attack_num, pkt_noise_num, \
               bytes_attack_reached_one, bytes_noise_reached_one, \
               pkt_drop_num, attack_ack

        """
        The defender action: {drop / not drop}
        - 目标：drop掉attacker的packets，一种policy可能是drop掉很多packets的ip_src
        - Choose a specific volume of packets to drop, to maintain the utilization in normal range.
        - Action in [.0, 0.1, 0.2, 0.3, 0.4] = Discrete(5)
            :params packets_items: In shape of src_ip, bytes, flow_id (0-user, 1-attacker)
            :return: the number of packet dropped in this step
            :return drop_attack - Indicting dropping packets from attack
            :return 
        """

    def defend_util(self, packets_items, pkt_attack_num, pkt_noise_num, action):
        bytes_attack_reached_one = 0.0
        bytes_noise_reached_one = 0.0

        pkt_attack_pass = 0
        pkt_noise_pass = 0
        attack_ack = defaultdict(int)  # How many packets passed by defender, which equals to ACK to attacker
        drop_dict = OrderedDict()  # How many packets need to be dropped
        defend_dict = defaultdict(list)  # A dict for all in-defend records

        # defend_drop_volume = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        # defend_drop_volume = [0.0, 0.1, 0.2, 0.3, 0.4]
        defend_drop_volume = [0.0, 0.2, 0.4, 0.6, 0.8]
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

        # 开始Drop，使用drop_dict对应的drop比例对每个ip drop
        # 统计被drop的packet信息
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
        # print("Getting attack_ack {}".format(attack_ack))

        pkt_attack_num = pkt_attack_pass
        pkt_noise_num = pkt_noise_pass
        return pkt_attack_num, pkt_noise_num, \
               bytes_attack_reached_one, bytes_noise_reached_one, \
               pkt_drop_num, attack_ack

    """
    Background noise
    Genrating the pkt of each user host, and aggregate the noise pkt
    !!! How to be Gaussian Noise???
        :return: the number of total noise pkt in this step
    """

    def user_noise(self):
        pkt_noise_num = 0
        # reset中已经连接好user的pg，直接读下一步timestep中产生的数据即可，
        # 这样background noise就符合Possion分布了
        for i in self.user_pg_list:
            # print("Getting User Noise")
            pkt_noise_num += i.packets_sent
            self.user.update_info(i.id, i.packets_sent)
            i.packets_sent = 0

        # if pkt_noise_num > self.pkt_noise_num:
        #     pkt_noise_num_step = pkt_noise_num - self.pkt_noise_num
        # else:
        #     pkt_noise_num_step = pkt_noise_num
        return pkt_noise_num

    def calculate_pkt_reached(self, ):
        pass

    def calculate_utilization_one_step(self, pkt_attack_num, pkt_drop_num, pkt_noise_num, bytes_attack_reached_one,
                                       bytes_noise_reached_one):
        """
        - Current throughput = current pkt/time_one_step (x)
        - utilization = bits /upper_bandwidth * time
        """
        cur_util = ((bytes_attack_reached_one + bytes_noise_reached_one) * 8) \
                   / (self.upper_bandwidth * self.units * self.time_one_step)
        # ONLY for Q-learning, finite states?
        self.utilization = cur_util
        return self.utilization
        # pass

    """
    calculate_utilization_culmative
    - Calculate all the packets in the network at this step

    """

    def calculate_utilization_atacker_culmative(self, ):
        # For botnet pkt rate
        total_pkt = 0
        for i, host in enumerate(self.botnet_array):
            total_pkt += host.pkt_sending
        return total_pkt
