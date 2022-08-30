"""
Abstract class to represent the Defender firewall

"""

from numpy import float64
from gym_cyberwargame.network_sim.host import hostObj
from random import expovariate
import simpy
from gym_cyberwargame.network_sim.SimComponents import PacketGenerator, PacketSink,SwitchPort_Defend,SwitchPort



class Defender():
    """
    Defender class: a switch 
    """
    def __init__(self, env, rate, qlimit=None, limit_bytes=True, debug=False):
        self.defender = SwitchPort_Defend(env, rate, qlimit=qlimit, limit_bytes=limit_bytes, debug=debug)
        self.name = "Defender"
        # return self.defender

    def get_defender(self,):
        return self.defender

    def __str__(self) -> str:

        return "Name: {}, rate: {}".\
            format(self.name, self.defender.rate)
