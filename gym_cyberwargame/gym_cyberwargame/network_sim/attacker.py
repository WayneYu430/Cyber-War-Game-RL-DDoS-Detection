"""
Abstract class to represent the Attacker bonet


"""

from typing import List
from numpy import float64
from gym_cyberwargame.network_sim.host import hostAttackObj, hostObj
from random import expovariate
import simpy
from gym_cyberwargame.network_sim.SimComponents import PacketGenerator, PacketSink


class Attacker():
    """
    Attacker Botnet List, contain hostObj to hold the IP_Src information 
        :param botnet_src_ip_addr: ip addr of the host list
    """
    def __init__(self, botnet_src_ip_addr, time_step):
        self.botnet_src_ip_addr = botnet_src_ip_addr
        self.botnet_array = []
        for i in self.botnet_src_ip_addr:
            tmp_host = hostAttackObj(i, 0.0, 0.0, 0.0)  # src_ip, pkt_sending, pkt_rate, time_interval
            self.botnet_array.append(tmp_host)

        self.time_step = time_step

    """
    attack_pg_list
        :return pg object to genertor attack packets in the time step
    """
    def attack_pg_list(self, envSim, chosen_ip, time_interval, sdist, flow_id="attacker") -> List:
        pg_list = []

        self.update_host_info(chosen_ip, time_interval)
        for i in self.botnet_array:
            if i.time_interval != 0:
                pg_tmp = self.attack_pg(envSim, i.ip_addr, i.time_interval, sdist, flow_id)
                pg_list.append(pg_tmp)
        # print("Generating the attacker list {}".format(pg_list))
        return pg_list
    """
    show_attack_pg_list
        :return str to show the status of the botnet
        and a hook for future render in graph
    """
    def show_attack_pg_list(self):
        return_str = ''
        # for i in self.botnet_array:
        return_str += '\n'.join(map(str, self.botnet_array))
        return return_str
    """
    update_host_info
    - Each step, chose an ip, and updaing the info store at the hostAttackObj
    - If no ip are in the botnet, raise ValueError
    """
    def update_host_info(self, chosen_ip, time_interval):
        updated = False
        for i in self.botnet_array:
            if i.ip_addr == chosen_ip:
                i.time_interval = time_interval
                updated = True
        if not updated:
            raise ValueError("Chosen IP is not in the botnet list")
        else:
            return 
    
    def attack_pg(self, envSim, chosen_ip, time_interval, sdist, flow_id):
        # time_interval_tmp = lambda x=scalr_interval:x
        """
        # Time_interval = 1s
        # 1000 packets/sec is 1/1000
        # [time_interval/5000, time_interval_upper/5000] is [5000 pkt/sec,1000 pkt/sec]
        """
        # print("In atacker_pg, getting adist {}".format(2000/time_interval))
        def adist():

            return expovariate(5000/time_interval)

        def distSize():
            return expovariate(sdist/1000)  # timse_step/1000 = 1s/1000 = 1000Bytes/pkt

        pg = PacketGenerator(env=envSim, id=chosen_ip, adist=adist, sdist=distSize, flow_id=flow_id,\
             finish=envSim.now+self.time_step)
        return pg

    def __str__(self) -> str:
        tmp=""
        for i in self.botnet_array:
            tmp += str(i)+'\n'
        return tmp
