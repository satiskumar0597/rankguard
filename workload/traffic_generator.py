"""
RankGuard: Workload traffic generator.
Generates mixed data-centre traffic: mice/elephant flows, incast,
congestion migration, all-to-all, stride, heavy-tailed ON/OFF.
"""

import random
import time
import subprocess
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum


class WorkloadType(Enum):
    PERIODIC       = 'periodic'
    ELEPHANT_MICE  = 'elephant_mice'
    INCAST         = 'incast'
    MIGRATION      = 'migration'
    ALL_TO_ALL     = 'all_to_all'
    STRIDE         = 'stride'
    HEAVY_TAIL     = 'heavy_tail'


@dataclass
class FlowSpec:
    src: str
    dst: str
    size_bytes: int
    start_time: float
    flow_type: str = 'mice'


class WorkloadGenerator:
    """
    Generates mixed workloads for RankGuard evaluation.
    Flow sizes follow a heavy-tailed distribution (Pareto).
    """

    # Pareto shape parameter: alpha=1.2 gives heavy tail
    PARETO_ALPHA   = 1.2
    MICE_THRESHOLD = 100_000       # bytes — flows below this are mice
    ELEPHANT_THRESHOLD = 10_000_000  # bytes — flows above this are elephants

    # ON/OFF parameters (seconds)
    ON_MEAN  = 2.0
    OFF_MEAN = 1.0

    def __init__(self, hosts: List[str], seed: int = 42):
        self.hosts = hosts
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    # ── flow size distribution ──────────────────────────────────────────────

    def _flow_size(self) -> int:
        """Heavy-tailed Pareto flow size in bytes."""
        raw = (self.np_rng.pareto(self.PARETO_ALPHA) + 1) * self.MICE_THRESHOLD
        return max(1024, int(raw))

    def _classify(self, size: int) -> str:
        if size < self.MICE_THRESHOLD:
            return 'mice'
        if size > self.ELEPHANT_THRESHOLD:
            return 'elephant'
        return 'medium'

    # ── workload builders ───────────────────────────────────────────────────

    def periodic_service(self, duration: float, rate: float = 10.0) -> List[FlowSpec]:
        """Regular background service-to-service traffic."""
        flows = []
        t = 0.0
        while t < duration:
            src, dst = self._pair()
            size = self._flow_size()
            flows.append(FlowSpec(src, dst, size, t, self._classify(size)))
            t += self.rng.expovariate(rate)
        return flows

    def elephant_mice_mix(self, duration: float,
                          elephant_fraction: float = 0.2) -> List[FlowSpec]:
        """Mixed elephant and mice flows."""
        flows = []
        t = 0.0
        while t < duration:
            src, dst = self._pair()
            if self.rng.random() < elephant_fraction:
                size = int(self.np_rng.pareto(0.8) * self.ELEPHANT_THRESHOLD)
                size = max(self.ELEPHANT_THRESHOLD, size)
                ftype = 'elephant'
            else:
                size = self.rng.randint(1024, self.MICE_THRESHOLD)
                ftype = 'mice'
            flows.append(FlowSpec(src, dst, size, t, ftype))
            t += self.rng.expovariate(50.0)
        return flows

    def incast(self, duration: float, fanin: int = 8,
               interval: float = 2.0) -> List[FlowSpec]:
        """Periodic incast bursts: many senders to one receiver."""
        flows = []
        t = interval
        while t < duration:
            dst = self.rng.choice(self.hosts)
            senders = self.rng.sample(
                [h for h in self.hosts if h != dst],
                min(fanin, len(self.hosts) - 1)
            )
            for src in senders:
                size = self.rng.randint(500_000, 5_000_000)
                flows.append(FlowSpec(src, dst, size, t, 'incast'))
            t += interval + self.rng.gauss(0, 0.1)
        return flows

    def congestion_migration(self, duration: float,
                             shift_at: float = None) -> List[FlowSpec]:
        """
        Hot-spot that migrates at shift_at seconds.
        First half: traffic concentrated on first quarter of hosts.
        Second half: traffic shifts to second quarter.
        """
        if shift_at is None:
            shift_at = duration / 2
        hot_a = self.hosts[:max(1, len(self.hosts) // 4)]
        hot_b = self.hosts[len(self.hosts) // 4: len(self.hosts) // 2]
        flows = []
        t = 0.0
        while t < duration:
            pool = hot_a if t < shift_at else hot_b
            src = self.rng.choice(pool)
            dst = self.rng.choice([h for h in self.hosts if h != src])
            size = self._flow_size()
            flows.append(FlowSpec(src, dst, size, t, self._classify(size)))
            t += self.rng.expovariate(20.0)
        return flows

    def all_to_all(self, duration: float,
                   rounds: int = 5) -> List[FlowSpec]:
        """All-to-all exchange rounds."""
        flows = []
        interval = duration / rounds
        for r in range(rounds):
            base_t = r * interval
            srcs = list(self.hosts)
            self.rng.shuffle(srcs)
            dsts = list(self.hosts)
            self.rng.shuffle(dsts)
            for src, dst in zip(srcs, dsts):
                if src == dst:
                    continue
                size = self.rng.randint(1_000_000, 100_000_000)
                t = base_t + self.rng.uniform(0, interval * 0.1)
                flows.append(FlowSpec(src, dst, size, t, 'elephant'))
        return flows

    def stride(self, duration: float) -> List[FlowSpec]:
        """Stride traffic: host i sends to host (i + n/2) % n."""
        n = len(self.hosts)
        flows = []
        t = 0.0
        while t < duration:
            for i, src in enumerate(self.hosts):
                dst = self.hosts[(i + n // 2) % n]
                size = self._flow_size()
                flows.append(FlowSpec(src, dst, size, t, self._classify(size)))
            t += self.rng.expovariate(2.0)
        return flows

    def heavy_tail_onoff(self, duration: float) -> List[FlowSpec]:
        """Heavy-tailed ON/OFF traffic."""
        flows = []
        for src in self.hosts:
            t = 0.0
            on = True
            while t < duration:
                if on:
                    on_dur = self.np_rng.exponential(self.ON_MEAN)
                    end = min(t + on_dur, duration)
                    while t < end:
                        dst = self.rng.choice(
                            [h for h in self.hosts if h != src]
                        )
                        size = self._flow_size()
                        flows.append(FlowSpec(src, dst, size, t,
                                              self._classify(size)))
                        t += self.rng.expovariate(5.0)
                else:
                    t += self.np_rng.exponential(self.OFF_MEAN)
                on = not on
        return flows

    def mixed(self, duration: float) -> List[FlowSpec]:
        """Full mixed workload used in the main evaluation."""
        all_flows = (
            self.periodic_service(duration, rate=5.0) +
            self.elephant_mice_mix(duration) +
            self.incast(duration) +
            self.congestion_migration(duration) +
            self.all_to_all(duration) +
            self.stride(duration) +
            self.heavy_tail_onoff(duration)
        )
        all_flows.sort(key=lambda f: f.start_time)
        return all_flows

    # ── helpers ─────────────────────────────────────────────────────────────

    def _pair(self) -> Tuple[str, str]:
        src = self.rng.choice(self.hosts)
        dst = self.rng.choice([h for h in self.hosts if h != src])
        return src, dst


def replay_flows(flows: List[FlowSpec], base_port: int = 5000,
                 duration: float = 1800.0):
    """
    Replay a flow list using iperf3.
    Each flow is launched as a background subprocess at its start_time.
    """
    procs = []
    t0 = time.time()
    for flow in flows:
        now = time.time() - t0
        wait = flow.start_time - now
        if wait > 0:
            time.sleep(wait)
        if time.time() - t0 > duration:
            break
        size_kb = max(1, flow.size_bytes // 1024)
        cmd = [
            'iperf3', '-c', flow.dst,
            '-n', f'{size_kb}K',
            '-p', str(base_port),
            '--logfile', '/dev/null'
        ]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description='RankGuard workload generator')
    parser.add_argument('--hosts', nargs='+', default=[f'10.0.0.{i}' for i in range(1, 17)])
    parser.add_argument('--workload', type=str, default='mixed',
                        choices=[w.value for w in WorkloadType])
    parser.add_argument('--duration', type=float, default=1800.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='flows.json')
    args = parser.parse_args()

    gen = WorkloadGenerator(hosts=args.hosts, seed=args.seed)
    fn = getattr(gen, args.workload if args.workload != 'mixed' else 'mixed')
    flows = fn(args.duration)

    with open(args.out, 'w') as f:
        json.dump([
            {'src': fl.src, 'dst': fl.dst,
             'size_bytes': fl.size_bytes,
             'start_time': fl.start_time,
             'flow_type': fl.flow_type}
            for fl in flows
        ], f, indent=2)
    print(f'Generated {len(flows)} flows → {args.out}')
