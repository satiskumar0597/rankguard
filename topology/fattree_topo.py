"""
RankGuard: Fat-tree / Clos topology generator for Mininet.
Supports k=4 (16 hosts), k=6 (54 hosts), k=8 (128 hosts).
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel


class FatTreeTopo(Topo):
    """
    Fat-tree topology with k pods.
      - k/2 core switches
      - k aggregation switches per pod  (k pods total)
      - k/2 edge switches per pod
      - k/2 hosts per edge switch
    Total hosts: (k^3)/4
    """

    def __init__(self, k=4, bw=1000, delay='0.1ms', **kwargs):
        self.k = k
        self.bw = bw          # Mbps
        self.delay = delay
        super(FatTreeTopo, self).__init__(**kwargs)

    def build(self):
        k = self.k
        half_k = k // 2

        # Core switches: (k/2)^2
        core = {}
        for i in range(half_k):
            for j in range(half_k):
                name = f'c_{i}_{j}'
                core[(i, j)] = self.addSwitch(name)

        agg = {}
        edge = {}
        hosts = {}

        for pod in range(k):
            # Aggregation switches
            for sw in range(half_k):
                name = f'a_{pod}_{sw}'
                agg[(pod, sw)] = self.addSwitch(name)

            # Edge switches
            for sw in range(half_k):
                name = f'e_{pod}_{sw}'
                edge[(pod, sw)] = self.addSwitch(name)

            # Hosts
            for sw in range(half_k):
                for h in range(half_k):
                    hname = f'h_{pod}_{sw}_{h}'
                    hosts[(pod, sw, h)] = self.addHost(
                        hname,
                        ip=f'10.{pod}.{sw}.{h + 2}/24'
                    )
                    self.addLink(
                        hosts[(pod, sw, h)],
                        edge[(pod, sw)],
                        bw=self.bw, delay=self.delay,
                        cls=TCLink
                    )

            # Edge to aggregation links
            for esw in range(half_k):
                for asw in range(half_k):
                    self.addLink(
                        edge[(pod, esw)],
                        agg[(pod, asw)],
                        bw=self.bw, delay=self.delay,
                        cls=TCLink
                    )

            # Aggregation to core links
            for asw in range(half_k):
                for j in range(half_k):
                    self.addLink(
                        agg[(pod, asw)],
                        core[(asw, j)],
                        bw=self.bw, delay=self.delay,
                        cls=TCLink
                    )

    def host_count(self):
        return (self.k ** 3) // 4


def build_topology(k=4, controller_ip='127.0.0.1', controller_port=6653):
    """Build and return a Mininet network with remote ONOS controller."""
    setLogLevel('info')
    topo = FatTreeTopo(k=k)
    net = Mininet(
        topo=topo,
        controller=RemoteController(
            'onos', ip=controller_ip, port=controller_port
        ),
        link=TCLink,
        autoSetMacs=True
    )
    return net


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RankGuard fat-tree topology')
    parser.add_argument('--k', type=int, default=4, choices=[4, 6, 8],
                        help='Fat-tree parameter k (4=16h, 6=54h, 8=128h)')
    parser.add_argument('--controller', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6653)
    args = parser.parse_args()

    net = build_topology(k=args.k,
                         controller_ip=args.controller,
                         controller_port=args.port)
    net.start()
    print(f'Fat-tree k={args.k}: {FatTreeTopo(k=args.k).host_count()} hosts')
    from mininet.cli import CLI
    CLI(net)
    net.stop()
