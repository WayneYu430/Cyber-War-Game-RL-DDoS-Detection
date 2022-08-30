"""
Abstract class to represent a host in the network env
"""

from numpy import float64


class hostObj():
    """
    Host object, contains ip_addr and its pkt sending rate currently
        :param ip_addr: ip addr of the host
        :param pkt_sending: number of pkt sending of host
        :param pkt_rate: rate of pkt sending of host = [0, 1]
    """
    def __init__(self, ip_addr: int, pkt_sending: float64, pkt_rate: float64):
        self.ip_addr = ip_addr
        self.pkt_sending = pkt_sending
        self.pkt_rate = pkt_rate
    def __str__(self) -> str:
        tmp = 'ip: 192.168.1.' + str(self.ip_addr).ljust(8) + 'pkt sending: ' + str(self.pkt_sending) +\
            ' in ' +str(self.pkt_rate)
        return tmp


class hostAttackObj(hostObj):
    def __init__(self, ip_addr: int, pkt_sending: float64, pkt_rate: float64, time_interval:float64, ):
        super().__init__(ip_addr, pkt_sending, pkt_rate)
        self.time_interval = time_interval

    def __str__(self) -> str:
        tmp = 'Attacker Host-ip: 192.168.1.' + str(self.ip_addr).ljust(8) + 'pkt sending: ' + str(self.pkt_sending) +\
            ' time inteval ' + str(self.time_interval)
        return tmp