"""Offline-to-online O(1) certified gain Look-Up Table (LUT) benchmark.

Implements uniform quantization, Flat Index Buffer, Nearest Neighbor,
and Trilinear Interpolation over the 3D operating grid (tau_c, tau_e, delta_max).
"""

from __future__ import annotations

import time
import numpy as np


class CertifiedGainLUT:
    """Compact 3D LUT storing certified (kp, kv, ka) gains in float32."""

    def __init__(
        self,
        tau_c_grid: np.ndarray,
        tau_e_grid: np.ndarray,
        delta_grid: np.ndarray,
        gains: np.ndarray,
    ) -> None:
        self.tc = np.asarray(tau_c_grid, dtype=np.float32)
        self.te = np.asarray(tau_e_grid, dtype=np.float32)
        self.dm = np.asarray(delta_grid, dtype=np.float32)
        # gains shape: (Nc, Ne, Nd, 3) float32
        self.gains = np.asarray(gains, dtype=np.float32)
        self.nc, self.ne, self.nd, _ = self.gains.shape
        self.flat = self.gains.reshape(-1, 3)
        self.tc_min, self.dtc = float(self.tc[0]), float(self.tc[1] - self.tc[0])
        self.te_min, self.dte = float(self.te[0]), float(self.te[1] - self.te[0])
        self.dm_min, self.ddm = float(self.dm[0]), float(self.dm[1] - self.dm[0])

    @property
    def memory_bytes(self) -> int:
        return int(self.gains.nbytes)

    def query_flat(self, tc: float, te: float, dm: float) -> np.ndarray:
        ic = min(max(int((tc - self.tc_min) / self.dtc), 0), self.nc - 1)
        ie = min(max(int((te - self.te_min) / self.dte), 0), self.ne - 1)
        id_ = min(max(int((dm - self.dm_min) / self.ddm), 0), self.nd - 1)
        return self.flat[ic * self.ne * self.nd + ie * self.nd + id_]

    def query_nearest(self, tc: float, te: float, dm: float) -> np.ndarray:
        ic = min(max(int(round((tc - self.tc_min) / self.dtc)), 0), self.nc - 1)
        ie = min(max(int(round((te - self.te_min) / self.dte)), 0), self.ne - 1)
        id_ = min(max(int(round((dm - self.dm_min) / self.ddm)), 0), self.nd - 1)
        return self.gains[ic, ie, id_]

    def query_trilinear(self, tc: float, te: float, dm: float) -> np.ndarray:
        fc = min(max((tc - self.tc_min) / self.dtc, 0.0), self.nc - 1.00001)
        fe = min(max((te - self.te_min) / self.dte, 0.0), self.ne - 1.00001)
        fd = min(max((dm - self.dm_min) / self.ddm, 0.0), self.nd - 1.00001)
        ic, ie, id_ = int(fc), int(fe), int(fd)
        wc, we, wd = fc - ic, fe - ie, fd - id_
        c00 = self.gains[ic, ie, id_] * (1.0 - wd) + self.gains[ic, ie, id_ + 1] * wd
        c01 = self.gains[ic, ie + 1, id_] * (1.0 - wd) + self.gains[ic, ie + 1, id_ + 1] * wd
        c10 = self.gains[ic + 1, ie, id_] * (1.0 - wd) + self.gains[ic + 1, ie, id_ + 1] * wd
        c11 = self.gains[ic + 1, ie + 1, id_] * (1.0 - wd) + self.gains[ic + 1, ie + 1, id_ + 1] * wd
        c0 = c00 * (1.0 - we) + c01 * we
        c1 = c10 * (1.0 - we) + c11 * we
        return c0 * (1.0 - wc) + c1 * wc


def benchmark_lut(n_queries: int = 20000, seed: int = 42) -> dict:
    """Benchmark 48-cell (4 x 4 x 3) certified gain LUT latency and memory footprint."""
    tc_grid = np.linspace(0.05, 0.20, 4)
    te_grid = np.linspace(0.20, 0.50, 4)
    dm_grid = np.linspace(0.02, 0.08, 3)
    gains = np.zeros((4, 4, 3, 3), dtype=np.float32)
    gains[..., 0] = 1.15
    gains[..., 1] = 1.18
    gains[..., 2] = 0.80
    lut = CertifiedGainLUT(tc_grid, te_grid, dm_grid, gains)

    rng = np.random.default_rng(seed)
    q_tc = rng.uniform(0.05, 0.20, n_queries)
    q_te = rng.uniform(0.20, 0.50, n_queries)
    q_dm = rng.uniform(0.02, 0.08, n_queries)

    results = {"grid_points": int(gains.size // 3), "memory_bytes": lut.memory_bytes, "memory_kb": round(lut.memory_bytes / 1024.0, 2)}
    for name, fn in [
        ("flat_index_buffer", lut.query_flat),
        ("nearest_neighbor", lut.query_nearest),
        ("trilinear_interpolation", lut.query_trilinear),
    ]:
        # Warmup
        for i in range(200):
            fn(float(q_tc[i]), float(q_te[i]), float(q_dm[i]))
        lat_us = np.empty(n_queries, dtype=np.float64)
        for i in range(n_queries):
            t0 = time.perf_counter_ns()
            fn(float(q_tc[i]), float(q_te[i]), float(q_dm[i]))
            lat_us[i] = (time.perf_counter_ns() - t0) * 1e-3
        results[name] = {
            "mean_us": round(float(np.mean(lat_us)), 2),
            "p99_us": round(float(np.percentile(lat_us, 99)), 2),
        }
    return results
