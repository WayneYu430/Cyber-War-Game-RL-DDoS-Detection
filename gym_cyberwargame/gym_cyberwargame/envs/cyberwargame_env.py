"""
RL environment for a cyber wargame Markov game
"""
from ast import arg
from collections import defaultdict
from msilib.schema import Error
import random
# from random import random
import gym
from gym import spaces
from matplotlib import units
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



class CyberWarGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    env_args={
        "pkt_attack_reached_num":   0,
        "pkt_noise_reached_num":    0,
        "bytes_noise_reached":      0.0,
        "bytes_attack_reached":     0.0,
        "botnet_src_ip_addr":       None,
        "user_src_ip_addr":         None,
        "server_ip_addr":           0,
        "bot_net_num":              10,
        "user_num":                 5,
        "utilization":              0.0,
        "interval_num":             100,
        "thresh_utilization":       0.8,
        "units":                    100000, # 1000000000
        "pkt_size":                 64,
        "pkt_per_sec":              10,
        "upper_bandwidth":          10,
        "reward_lambda":            0.6,
        "reward":                   0,
        "time_one_step":            10,
        "save_dir":                 None,
        "initial_state_path":       None,
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
        self.server_host = hostObj(0, 0.0, 0.0)               # ip_addr, pkt_sending,pkt_rate
        self.botnet_array = []                          # array of hostObj
        self.user_array = []                            # array of hostObj
        self.network_array = []                         # record of whole network
        self.botnet_src_ip_addr = env_args['botnet_src_ip_addr']
        self.user_src_ip_addr = env_args['user_src_ip_addr']
        self.server_ip_addr = env_args['server_ip_addr']
        self.bot_net_num = env_args['bot_net_num']
        self.user_num = env_args['user_num']

        # Config for network state
        self.utilization = env_args['utilization']
        self.interval_num = env_args['interval_num']                # 10 interval
        self.thresh_utilization = env_args['thresh_utilization']    # [0, 1]
        self.units = env_args['units']                              # scale units in 1000000000 = 10^9 = 1Gbps
        self.pkt_per_sec = env_args['pkt_per_sec']                  # 100000 pkt/s = 100 Kpkt/s = 6.4Mbps
        self.pkt_size = env_args['pkt_size']                        # 64 Bytes = 1 Pkt
        self.time_one_step = env_args['time_one_step']              # 10s 
        self.upper_bandwidth = env_args['upper_bandwidth']          # 10000000 bits/s = 1000 Mbps = 10 Gbps

        # Use for reward function
        self.reward_lambda = env_args['reward_lambda']
        self.reward = env_args['reward']
        
        # Use for two agent in Tianshou
        self.agent_idx = ['attacker', 'defender']
        self.agents = ['attacker', 'defender']
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
        tmp_ip_addr_array = rng.choice(255, size=1+self.bot_net_num+self.user_num, replace=False)
        self.server_ip_addr =  int(tmp_ip_addr_array[:1])
        self.botnet_src_ip_addr = tmp_ip_addr_array[1:1+self.bot_net_num]
        self.user_src_ip_addr = tmp_ip_addr_array[1+self.bot_net_num:]
        
        self.botnet = Attacker(self.botnet_src_ip_addr, self.time_one_step)
        self.botnet_array = self.botnet.botnet_array
        self.user = User(self.user_src_ip_addr, self.time_one_step)
        self.user_array = self.user.user_array

        # Use constant for now 0701
        self.defender = Defender(self.simpy_env, 10000000 ).get_defender()

        """
        Observation for Attacker:  {src_ip, pkt_sent, ACK (the pkt reached the server)}
            - Attack should only see the packets of botnet?
            - 1. If attack see the utilization (x)
            - 2. If attack only see the botnet information in a list:  {src_ip, pkt_sent, ACK(the pkt reached the server)} of botnet
        Observation for Defender: utilization, {src_ip_addr, pkt_sent} (array)
            - Defender see the server
            - Can defender see user??? - NO
        """
        
        self.observation_space = spaces.Dict(
            {  
                "attacker": self.__get_attack_obs(),
                # "attacker": spaces.Box(np.array([0, 0.0, 0]), np.array([np.inf, 1.0, 255]), ),
                # "defender": spaces.Box(np.array([0.0, 0]), np.array([1.0, 255]), shape=(2,), )
                "defender": self.__get_defend_obs()
            }
        )
    

        # For attacker, we have different numbers and differetn ip_scr number of attack action. thus. a MultiDiscrete
        # For defender, we have 2 action, {drop/ Not drop} the packet
        self.attacker_action_space = spaces.MultiDiscrete([self.interval_num, self.bot_net_num])  # (time, ip)
        # self.action_space = spaces.MultiDiscrete([self.interval_num, self.bot_net_num])
        # Use for RLlib (X)
        # self.action_space = spaces.Tuple([spaces.Discrete(self.interval_num), spaces.Discrete(self.bot_net_num)])
        
        # self.attacker_action_space = spaces.Discrete(self.interval_num * self.bot_net_num)
        self.defender_action_space = spaces.Discrete(2) # 0 drop and 1 not drop
        self.action_space = spaces.Dict({
            "attacker": self.attacker_action_space,
            "defender": self.defender_action_space
        })

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

    def reset(self, seed=None, return_info=False, options=None):    # return state
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        # super().reset()
        # Set the initial state
        self.utilization = 0.0
        self.pkt_reached_num  = 0.0
        self.pkt_attack_reached_num = 0
        self.pkt_noise_reached_num = 0
        self.pkt_noise_reached_num = 0
        self.bytes_noise_reached = 0
        self.bytes_attack_reached = 0
        self.simpy_env = simpy.Environment()
        self.defender = Defender(self.simpy_env, 10000000).get_defender()
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
        self.simpy_env.run(until=self.simpy_env.now+self.time_one_step)

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    """
    Chose an action (the index from action_space)
        :params action: a dict of agent_id & agent_action index 
        return (state, reward, done, info)
    """
    def step(self, action):
        # 0. Assign the multi action set, action is a dict
        attack_action = None
        defend_action = None
        if action is dict:
            for key, value in action.items(): 
                if key == "attack_action":
                    attack_action = np.array(value)
                    print("Get attack_action_value", type(value))
                elif key == "defend_action":
                    defend_action = value
        else:   # Only single action for attacker
            attack_action = action
            defend_action = 0


        # 1. Attack ation logic, action shape = (time, ip), time can not be 0!
        interval_num, src_ip_addr = self.interpret_attack_action(action=attack_action)
        # pkt_attack_num = self.attack(interval_num, src_ip_addr)
        attack_pg_list = self.attack(interval_num, src_ip_addr)
        for i in attack_pg_list:
            print("--Attack List of",str(i))
            i.out = self.defender

        # 2. Move a step of simpy_env, get user noist and attack packets
        self.simpy_env.run(until=self.simpy_env.now+self.time_one_step)
        # pkt_attack_num = self.calculate_utilization_atacker_culmative()

        # 3. Defend Logic, simple random choose a src to drop
        # Drop using items list return from the FIFO queue to iterate act drop or not drop
        # - Collect information of packets and bytes.
        # - Construct the observation for attack and defender
        packets_items = self.packet_sink.packet_items
        # print("Get packets in queue, len == {}".format(len(packets_items)))
        
        pkt_noise_num = self.user_noise()
        pkt_attack_num=0
        for i in attack_pg_list:
            pkt_attack_num += i.packets_sent
            # Updating the botnet info, for obs construction
            for attack_host in self.botnet_array:
                if i.id==attack_host.ip_addr:
                    attack_host.pkt_sending = i.packets_sent

        print("1. Before Dropping: attack sent {}, User sent {}".format(pkt_attack_num, pkt_noise_num))

        # A random defender policy
        pkt_attack_num, pkt_noise_num,\
            bytes_attack_reached_one, bytes_noise_reached_one, \
                pkt_drop_num, attack_drop= self.defend(packets_items, pkt_attack_num, pkt_noise_num)

        print("2. After Dropping:  attack reached {}, user reached {}, total dropped: {}".format(pkt_attack_num, pkt_noise_num, pkt_drop_num))

        self.pkt_attack_reached_num += pkt_attack_num
        self.pkt_noise_reached_num += pkt_noise_num
        self.bytes_attack_reached += bytes_attack_reached_one
        self.bytes_noise_reached += bytes_noise_reached_one

        # Updating the hostObj array of user, botnet
        # Updating attacker
        print("Attacker ACK Reaching\n",attack_drop)    # Number of packets reached equal to ACK reply, as observation

        total_pkt = self.packet_sink.packets_rec
        


        # 4. Calculate utlization 
        cur_util = self.calculate_utilization_one_step(pkt_attack_num, pkt_drop_num, pkt_noise_num\
            ,self.bytes_attack_reached, self.bytes_noise_reached)

        print("cur_util=%s" % str(cur_util))
        # . Done logic
        if cur_util > self.thresh_utilization:  # will done???
            done = True
        else:
            done = False
        # total_pkt = pkt_attack_num-pkt_drop_num+pkt_noise_num
        
        # print("Total packet is ", total_pkt)
        
        if done:
            self.reward = 10  # indicate attacker win
        else:
            prob_attack = 0.0
            prob_user = 0.0
            prob_attack = pkt_attack_num/total_pkt  # How many attack packets reached the server
            prob_user = pkt_noise_num/total_pkt     # How many user packets reached the server

            self.reward = self.reward_lambda * prob_attack - (1-self.reward_lambda)*(1-prob_user)
            print("===== The prob_attack is %f , The prob_user is %f" %(prob_attack, prob_user))
            print("===== The reward is ", self.reward)
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, self.reward, done, info

    def render(self, mode="human"):
        pass

    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    """
    ================================ Private Method ================================
    """

    """
    """
    def _get_obs(self, attack_ack=None):
        # For attack
        attack_obs = defaultdict(int)
        attack_list = []
        if attack_ack is None:
            for i in self.botnet_array:
                attack_obs[i.ip_addr] = [i.ip_addr, i.pkt_sending, 0]   # {src_ip, pkt_sent, ACK (the pkt reached the server)}
                attack_list.append([i.ip_addr, i.pkt_sending, 0]) 
        else:
            for i in self.botnet_array:
                attack_obs[i.ip_addr] = [i.ip_addr, i.pkt_sending, attack_ack[i.ip_addr]]   # {src_ip, pkt_sent, ACK (the pkt reached the server)}
                attack_list.append([i.ip_addr, i.pkt_sending, attack_ack[i.ip_addr]])
        
        # For defender
        defender_obs = {}
        defender_obs['utilization'] = self.utilization
        net_info = {}
        for i in self.botnet_array:
            net_info[i.ip_addr] = [i.ip_addr, i.pkt_sending]
        for i in self.user_array:
            net_info[i.ip_addr] = [i.ip_addr, i.pkt_sending]
        defender_obs['net_info'] = net_info

        return {"attacker": attack_obs, "defender": defender_obs}
        # print("received: {}, attack sent {}, User sent {}".format(self.defender.packets_rec, pkt_attack_num, tmp_user_pkt))


    """
    Get the environment info at the begining
        
    """
    def _get_info(self):
        bot_net_str = '; \n'.join("attack-"+str(x) for x in self.botnet_array)
        user_str = '; \n'.join("user  -"+str(x) for x in self.user_array)
        server_str = "server-"+str(self.server_host)
        return bot_net_str+'\n'+user_str+'\n'+server_str

    """
    attack_obs={src_ip, pkt_sent, ACK (the pkt reached the server)}

    """
    def __get_attack_obs(self):
        obs_dict = defaultdict(int)
        for i in self.botnet_array:
            obs_dict[i.ip_addr] = spaces.Box(low=np.array([0, 0, 0]), high=np.array([255, np.inf, np.inf]), \
                 shape=(3,), dtype=int)
            # obs_list.append(spaces.Box(low=np.array([0.0, 0]), high=np.array([0.0, 255]), shape=(2,), dtype=np.float64))
            # Regard IP as feature??? Box in 2D 
            # obs_dict[str(i.ip_addr)] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
        __attack_space = spaces.Dict(obs_dict)
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
        for i in self.botnet_array:
            net_info[i.ip_addr] = spaces.Box(low=np.array([0, 0]), high=np.array([255, np.inf]), \
                 shape=(2,), dtype=int)
        for i in self.user_array:
            net_info[i.ip_addr] = spaces.Box(low=np.array([0, 0]), high=np.array([255, np.inf]), \
                 shape=(2,), dtype=int)
        obs_dict['net_info'] = spaces.Dict(net_info) 

        _defend_space = spaces.Dict(obs_dict) 
        # print("__attack_space{}".format(obs_dict.items()))
        return _defend_space

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
        print(action)
        # if not isinstance(action, np.ndarray):
        #     print(type(action))
        #     raise ValueError("action is not ndarray")
        total_action = len(self.botnet_array)
        print("Type of action", type(action))
        interval_num = 1
        if interval_num == 0:
            raise ValueError("Interval Num can not be 0")
        src_ip_addr = 0
        interval_num = action[0]
        src_ip_addr = self.botnet_array[action[1]].ip_addr
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
    def attack(self, chosen_interval_num, src_ip_addr, size_dist = 0.1):

        attack_pg_list = self.botnet.attack_pg_list(self.simpy_env, src_ip_addr, chosen_interval_num, size_dist)
        return attack_pg_list
        # pass

    """
    The defender action: {drop / not drop}
    - Random choose a pkt-sending value, and drop to zero.
        :params 
        :return: the number of packet dropped in this step
        :return drop_attack - Indicting dropping packets from attack
    """
    def defend_old(self, action=0, src_ip_addr=0, pkt_drop_prob = 1):
        pkt_drop_prob = random.random()
        user_or_botnet = random.randint(0, 1)
        drop_pkt = 0
        drop_attack = False     
        if user_or_botnet==0:
            drop_attack = False
            random_ip = random.randint(0, len(self.user_array)-1)
            src_ip_addr = self.user_array[random_ip].ip_addr
            for i in self.user_array:
                # print("!!!!!!Updating self.user_array[i] ", i)
                if i.ip_addr == src_ip_addr:
                    drop_pkt = round(i.pkt_sending * pkt_drop_prob)
                    i.pkt_sending = i.pkt_sending - drop_pkt
        elif user_or_botnet==1:
            drop_attack = True
            random_ip = random.randint(0, len(self.botnet_array)-1)
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
        :return: the number of packets dropped in this step
    """
    def defend(self, packets_items, pkt_attack_num, pkt_noise_num, action=0):
        pkt_drop_num = 0
        bytes_attack_reached_one = 0.0
        bytes_noise_reached_one = 0.0
        attack_ack = defaultdict(int)
        for i in packets_items:     # src_ip, bytes, flow_id
            drop_flag = random.randint(0,1)
            if drop_flag == 1:
                pkt_drop_num += 1
                if i[2] == "attacker":
                    pkt_attack_num -= 1
                    if pkt_attack_num <= 0:
                        break
                elif i[2] == "user":
                    pkt_noise_num -= 1
                    if pkt_noise_num <= 0:
                        break
            elif drop_flag == 0:
                if i[2] == "attacker":
                    attack_ack[i[0]] += 1
                    bytes_attack_reached_one += float(i[1])
                else:
                    bytes_noise_reached_one += float(i[1])
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
        # for i, v in enumerate(self.user_array):
        #     random_pkt = random.randint(1*self.units, 10*self.units)
        #     pkt_noise_num += random_pkt
        #     self.user_array[i].pkt_sending = random_pkt
        # self.user_pg_list = self.user.noise_pg(self.simpy_env)
        # reset中已经连接好user的pg，直接读下一步timestep中产生的数据即可，
        # 这样background noise就符合Possion分布了
        for i in self.user_pg_list:
            # print("Getting User Noise")
            pkt_noise_num += i.packets_sent
        return pkt_noise_num

    def calculate_pkt_reached(self,):
        pass

    def calculate_utilization_one_step(self, pkt_attack_num, pkt_drop_num, pkt_noise_num, bytes_attack_reached_one, bytes_noise_reached_one):
        """
        - Current throughput = current pkt/time_one_step (x)
        - utilization = bits /upper_bandwidth * time 
        """
        cur_util = ((bytes_attack_reached_one+bytes_noise_reached_one)*8) / (self.upper_bandwidth*self.units*self.time_one_step)
        print("Current bytes for attack=%s, user=%s" %(bytes_attack_reached_one, bytes_noise_reached_one))
        # ONLY for Q-learning, finite states?
        self.utilization = cur_util
        return self.utilization
        # pass

    """
    calculate_utilization_culmative
    - Calculate all the packets in the network at this step

    """
    def calculate_utilization_atacker_culmative(self,):
        # For botnet pkt rate
        total_pkt = 0
        for i, host in enumerate(self.botnet_array):
            total_pkt += host.pkt_sending
        return total_pkt
    """
    ================================ Helper Functions \end ================================
    """


