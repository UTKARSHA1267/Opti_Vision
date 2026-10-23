"""Phase 3, Part A (2): MAPPO building blocks (PyTorch).

Actor   ONE network shared by all camera agents (parameter sharing). Input: an agent's observation.
        Output: two sets of choices (importance level, margin vote), each a categorical distribution.
Critic  "central": estimates the team's future reward from the central STATE (MAPPO: centralised training).
        "local":   estimates it from each agent's OWN observation (independent PPO, the ablation).
RunningNorm keeps a running mean/std of inputs so the networks see well-scaled numbers.
"""
import numpy as np
import torch
import torch.nn as nn

from ma_env import N_IMPORTANCE, N_MARGIN, OBS_DIM, STATE_DIM


def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


class Actor(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.head_importance = nn.Linear(hidden, N_IMPORTANCE)
        self.head_margin = nn.Linear(hidden, N_MARGIN)
        for head in (self.head_importance, self.head_margin):     # start close to uniform choices
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        h = self.body(x)
        return self.head_importance(h), self.head_margin(h)

    def dist(self, x):
        li, lm = self(x)
        return torch.distributions.Categorical(logits=li), torch.distributions.Categorical(logits=lm)


class Critic(nn.Module):
    def __init__(self, kind="central", hidden=128):
        super().__init__()
        self.kind = kind
        self.net = mlp(STATE_DIM if kind == "central" else OBS_DIM, hidden, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RunningNorm:
    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, len(self.mean))
        bm, bv, bc = x.mean(axis=0), x.var(axis=0), len(x)
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x):
        return np.clip((np.asarray(x) - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)

    def state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    @classmethod
    def from_state(cls, s):
        n = cls(len(s["mean"]))
        n.mean, n.var, n.count = np.asarray(s["mean"]), np.asarray(s["var"]), s["count"]
        return n


def save(path, actor, critic, obs_norm, state_norm, info):
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "critic_kind": critic.kind,
                "obs_norm": obs_norm.state(), "state_norm": state_norm.state(), "info": info}, path)


def load(path, hidden=128):
    d = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(hidden)
    actor.load_state_dict(d["actor"])
    actor.eval()
    return actor, RunningNorm.from_state(d["obs_norm"]), d
