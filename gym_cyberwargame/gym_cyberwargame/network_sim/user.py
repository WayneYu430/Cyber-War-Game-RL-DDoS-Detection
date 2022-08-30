"""
Abstract class to represent the Attacker bonet


"""

from numpy import float64
from gym_cyberwargame.network_sim.host import hostObj
from random import expovariate
import simpy
from gym_cyberwargame.network_sim.SimComponents import PacketGenerator, PacketSink



class User():
    """
    User Background List, contain hostObj to hold the IP_Src information 
        :param user_src_ip_addr: ip addr of the host list
    """
    def __init__(self, user_src_ip_addr, time_step, time_interval=1, size_dist=1):
        self.user_src_ip_addr = user_src_ip_addr
        self.user_array = []
        self.time_interval = time_interval
        self.size_dist = size_dist

        for i in self.user_src_ip_addr:
            tmp_host = hostObj(i, 0.0, 0.0)
            self.user_array.append(tmp_host)

        self.time_step = time_step

    """
    noise_pg
        :return pg object to genertor attack packets in the time step
    """
    def noise_pg(self, envSim, flow_id="user"):
        # ps = PacketSink(envSim, debug=True)
        user_pg_list = []
        # time_interval = lambda x=self.time_interval:x

        def aDist():
            return expovariate(1000/self.time_interval)    # Use self.time_interval=1000, which means 1000 packets /sec

        def distSize():
            return expovariate(self.size_dist/1000)     # timse_step/1000 = 1s/1000 = 1000Bytes/pkt

        for i in self.user_array:
            pg = PacketGenerator(env=envSim, id=i.ip_addr, adist=aDist, sdist=distSize, flow_id=flow_id) # finish=envSim.now+self.time_step
            # print(pg.initial_delay)
            user_pg_list.append(pg)
        # pg.out = ps
        # envSim.run(until=self.time_step+1)
        # print(ps.packets_rec)
        return user_pg_list

    """
        update_info
        - update the user noise information in user host array
    """
    def update_info(self, ip_addr, pkt_sent):
        for i in self.user_array:
            if i.ip_addr == ip_addr:
                i.pkt_sending = pkt_sent
    """
    show_user_noise
        :return str to show the status of the user hosts
        and a hook for future render in graph
    """
    def show_user_noise(self):
        return_str = ''
        # for i in self.botnet_array:
        return_str += '\n'.join(map(str, self.user_array))
        return return_str

    def __str__(self) -> str:
        tmp=""
        for i in self.user_array:
            tmp += str(i)+'\n'
        return tmp
