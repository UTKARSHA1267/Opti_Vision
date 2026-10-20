"""Probing: the environment extension, the estimator's inputs, and the probe-and-allocate policy.

ProbeEnv          MultiCameraEnv + a probe action per camera (charged to the budget, reveals probe_rd)
ProbeData         probe table, features and helper arrays, loaded once
make_rows         estimator inputs for (camera state, candidate setting), shared by training and policies
ErrorEstimator    (v2, kept for reference) LightGBM estimator
LinearProbeEstimator  (v3) simple per-setting estimator built on probe results
ProbeAllocate     probe on average `rate` cameras per segment (oldest information first), estimate
                  errors, allocate the budget with those estimates
"""
import _paths  # noqa: F401

import lightgbm as lgb
import numpy as np

from _paths import OUT2, RESULTS2
from allocation import allocate
from doubt_detector import candidate_table
from p2_config import PROBE_MAX_AGE, REL_WINDOW
from policies import Policy
from traffic_env import MultiCameraEnv

REL_IMG = [0, 2, 3]   # brightness, blur, motion_mean (relative to the camera's own recent past)

FEATURES = ["obs_count", "obs_count_std", "obs_conf", "obs_unsure", "obs_near", "obs_crowded",
            "obs_count_change", "rel_brightness", "rel_blur", "rel_motion", "obs_setting_cost",
            "probe_rd", "probe_age", "probe_base_cost",
            "cand_setting", "cand_cost", "cand_prior_err", "cand_model", "cand_imgsz", "cand_fps",
            "cand_same_base_as_probe"]


def past_window_mean(arr, stream_of_seg, window):
    """For each segment t: mean of arr over the previous `window` segments of the same stream."""
    out = np.full(arr.shape, np.nan)
    for s in np.unique(stream_of_seg):
        idx = np.flatnonzero(stream_of_seg == s)
        x = arr[idx]
        for k in range(1, len(idx)):
            out[idx[k]] = x[max(0, k - window):k].mean(axis=0)
    return out


def take(arr, idx):
    out = np.asarray(arr[np.maximum(idx, 0)], dtype=float).copy()
    out[np.asarray(idx) < 0] = np.nan
    return out


class ProbeData:
    def __init__(self, tab):
        p = np.load(OUT2 / "probe.npz")
        f = np.load(OUT2 / "features.npz")
        self.tab = tab
        self.probe_rd = p["probe_rd"]
        self.config_base = p["config_base"]
        self.base_frame_cost = p["base_frame_cost"]
        self.ref_base = int(p["ref_base"])
        self.probe_frames = int(p["probe_frames"])
        self.det = f["det"]
        self.img = f["img"]
        self.img_hist = past_window_mean(self.img, tab.stream_of_seg, REL_WINDOW)
        self.cand = candidate_table(tab)
        self.cheap_base = int(self.config_base[0])

    def probe_cost(self, setting=None):
        """Cost of one probe: cheapest model/resolution + reference, on probe_frames frames."""
        return self.probe_frames * (self.base_frame_cost[self.cheap_base] + self.base_frame_cost[self.ref_base])

    def make_rows(self, prev1, prev2, j, rd, age, pbase):
        """Inputs for every candidate setting. All arguments are arrays of length m; -1 / nan = unknown.
        prev1, prev2: table rows of the camera's previous two segments; j: setting run at prev1;
        rd, age, pbase: latest probe result, its age in segments, and the model/resolution probed."""
        prev1, prev2, j = (np.asarray(x, dtype=int) for x in (prev1, prev2, j))
        pbase = np.asarray(pbase, dtype=int)
        tab, C = self.tab, self.tab.n_configs
        jj = np.maximum(j, 0)
        d1 = self.det[np.maximum(prev1, 0), jj].astype(float)
        d1[(prev1 < 0) | (j < 0)] = np.nan
        d2 = self.det[np.maximum(prev2, 0), jj, 0].astype(float)
        d2[(prev2 < 0) | (j < 0)] = np.nan
        rel = take(self.img, prev1)[:, REL_IMG] - take(self.img_hist, prev1)[:, REL_IMG]
        jcost = take(tab.costs, j)
        pcost = take(self.base_frame_cost, pbase)
        age = np.where(np.asarray(age, dtype=float) > PROBE_MAX_AGE, np.nan, age)
        rd = np.where(np.isnan(age), np.nan, rd)
        base = np.column_stack([d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3], d1[:, 4], d1[:, 7], d1[:, 0] - d2,
                                rel, jcost, rd, age, pcost])
        X = np.repeat(base, C, axis=0)
        cf = np.tile(self.cand, (len(base), 1))
        same = (self.config_base[cf[:, 0].astype(int)] == np.repeat(pbase, C)).astype(float)
        return np.column_stack([X, cf, same])


class ErrorEstimator:
    def __init__(self, data, path=RESULTS2 / "estimator_v2" / "point.txt"):
        self.data = data
        self.booster = lgb.Booster(model_file=str(path))

    def predict(self, prev1, prev2, j, rd, age, pbase):
        X = self.data.make_rows(prev1, prev2, j, rd, age, pbase)
        return self.booster.predict(X).reshape(-1, self.data.tab.n_configs)


class LinearProbeEstimator:
    """predicted error of setting c = w0[c] + w1[c] * latest probe x exp(-(age-1)/tau)
                                   + w2[c] * mean of the camera's recent probes + w3[c] * (has any probe)
    Fitted per setting on the training split (estimator_v3.py). Cannot memorise cameras."""

    def __init__(self, path=RESULTS2 / "estimator_v3" / "coefficients.json"):
        import json
        d = json.load(open(path))
        self.w = np.array(d["weights"])          # (C, 4)
        self.tau = d["tau"]
        self.rd_fill = d["rd_mean_train"]

    @staticmethod
    def design(rd_last, age, rd_mean, tau, rd_fill):
        rd_last, age, rd_mean = (np.asarray(x, dtype=float) for x in (rd_last, age, rd_mean))
        has = ~np.isnan(rd_last)
        short = np.where(has, np.nan_to_num(rd_last) * np.exp(-(np.nan_to_num(age, nan=1) - 1) / tau), 0.0)
        long = np.where(np.isnan(rd_mean), rd_fill, rd_mean)
        return np.column_stack([np.ones(len(short)), short, long, has.astype(float)])

    def predict(self, rd_last, age, rd_mean):
        X = self.design(rd_last, age, rd_mean, self.tau, self.rd_fill)
        return X @ self.w.T                      # (n_cameras, C)


class ProbeEnv(MultiCameraEnv):
    """MultiCameraEnv + a probe action per camera: the cheapest model/resolution and the reference run on
    a few frames, their disagreement is returned, and the probe's GPU time is charged to the budget."""

    def __init__(self, table, data, **kw):
        super().__init__(table, **kw)
        self.data = data

    def step(self, proposal, probe=None, extra_cost=None, **kw):
        proposal = np.asarray(proposal, dtype=int)
        probe = np.zeros(self.n, dtype=bool) if probe is None else np.asarray(probe, dtype=bool)
        extra = np.zeros(self.n) if extra_cost is None else np.asarray(extra_cost, dtype=float)
        pc = self.data.probe_cost()
        extra = extra + np.where(probe, pc, 0.0)
        obs, reward, done, info = super().step(proposal, extra_cost=extra, **kw)
        rd = np.full(self.n, np.nan)
        rd[probe] = self.data.probe_rd[info["rows"][probe], self.data.cheap_base]
        info.update(probe=probe, probe_rd=rd, probe_ms=float(pc * probe.sum()))
        return obs, reward, done, info


class ProbeAllocate(Policy):
    """Probe on average `rate` cameras per segment (those with the oldest probe first), estimate every
    setting's error per camera from its probe history, then allocate the remaining budget."""

    def __init__(self, estimator, rate, long_window=30):
        self.est, self.rate, self.long_window = estimator, rate, long_window
        self.name = f"probe_allocate_r{rate:g}"

    def reset(self, env):
        self.history = [[] for _ in range(env.n)]   # (step, rd) per camera
        self.staleness = np.full(env.n, 1e9)
        self.credit = 0.0

    def _state(self, t):
        n = len(self.history)
        rd_last, age, rd_mean = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for i, h in enumerate(self.history):
            recent = [r for (s, r) in h if t - s <= self.long_window]
            if h:
                s, r = h[-1]
                rd_last[i], age[i] = r, t - s
            if recent:
                rd_mean[i] = float(np.mean(recent))
        return rd_last, age, rd_mean

    def act(self, obs, env):
        t = obs["t"]
        est = self.est.predict(*self._state(t))
        self.credit += self.rate
        k = int(min(np.floor(self.credit + 1e-9), env.n))
        self.credit -= k
        probe = np.zeros(env.n, dtype=bool)
        if k > 0:
            probe[np.lexsort((np.arange(env.n), -self.staleness))[:k]] = True
        cap = env.scheduler.limit(env.budget, env.backlog, reserved=float(env.data.probe_cost() * probe.sum()))
        return {"proposal": allocate(est, env.costs, cap, env.weights), "probe": probe, "err_est": est}

    def update(self, info, env):
        t = env.t - 1                               # the step just executed
        self.staleness += 1
        for i in np.flatnonzero(info["probe"]):
            self.history[i].append((t, info["probe_rd"][i]))
            self.staleness[i] = 0