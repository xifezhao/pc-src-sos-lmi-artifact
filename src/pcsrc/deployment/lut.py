"""Offline-to-online O(1) Dual-Certificate-verified 3D gain Look-Up Table (LUT).

Synthesizes real Dual-Certificate-verified (kp, kv, ka) gains at every cell of the
4 x 4 x 3 = 48-cell operating grid (tau_c, tau_e, delta_max), exports NPZ/CSV tables,
and benchmarks O(1) Flat Index Buffer, Nearest Neighbor, and Trilinear Interpolation
lookup latencies directly against the certified table.
"""

from __future__ import annotations

import csv
from pathlib import Path
import time

import numpy as np

from pcsrc.models.platoon import PlatoonParams


def _dual_cert_margin(
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    delta_max: float,
    w: np.ndarray,
) -> float:
    """Exact algebraic Dual-Certificate margin: min_w [sqrt(|D_nom|^2) - sqrt(Z1) - sqrt(N)]."""
    if (1.0 + kv * h) * (kp * h + kv) <= tau_e * kp + 1e-4:
        return -1.0
    x = w**2
    a0 = kp**2
    a1 = (kp * h + kv) ** 2 - 2.0 * kp * (1.0 + kv * h)
    a2 = (1.0 + kv * h) ** 2 - 2.0 * tau_e * (kp * h + kv)
    a3 = tau_e**2
    d_sq = a0 + a1 * x + a2 * x**2 + a3 * x**3
    z1 = (delta_max**2) * x * (kp**2 + kv**2 * x) * (1.0 + h**2 * x)
    if np.any(d_sq <= z1 + 1e-9):
        return -1.0
    s = 1j * w
    n_sq = np.abs(kp + kv * s + ka * (s**2) * np.exp(-tau_c * s)) ** 2
    return float(np.min(np.sqrt(d_sq) - np.sqrt(z1) - np.sqrt(n_sq)))


class CertifiedGainLUT:
    """Compact 3D LUT storing Dual-Certificate-verified (kp, kv, ka) gains in float32."""

    def __init__(
        self,
        tau_c_grid: np.ndarray,
        tau_e_grid: np.ndarray,
        delta_grid: np.ndarray,
        gains: np.ndarray,
        margins: np.ndarray | None = None,
    ) -> None:
        self.tc = np.asarray(tau_c_grid, dtype=np.float32)
        self.te = np.asarray(tau_e_grid, dtype=np.float32)
        self.dm = np.asarray(delta_grid, dtype=np.float32)
        # gains shape: (Nc, Ne, Nd, 3) float32
        self.gains = np.asarray(gains, dtype=np.float32)
        self.margins = (
            np.asarray(margins, dtype=np.float32)
            if margins is not None
            else np.zeros(self.gains.shape[:3], dtype=np.float32)
        )
        self.nc, self.ne, self.nd, _ = self.gains.shape
        self.flat = self.gains.reshape(-1, 3)
        self.tc_min, self.dtc = float(self.tc[0]), float(self.tc[1] - self.tc[0])
        self.te_min, self.dte = float(self.te[0]), float(self.te[1] - self.te[0])
        self.dm_min, self.ddm = float(self.dm[0]), float(self.dm[1] - self.dm[0])

    @property
    def memory_bytes(self) -> int:
        return int(self.gains.nbytes)

    @property
    def certified_cells(self) -> int:
        return int(np.sum(self.margins > 0.0))

    def save(self, out_dir: Path, stem: str = "certified_lut_4x4x3") -> tuple[Path, Path]:
        """Export certified LUT to both binary NPZ and human-readable CSV."""
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"{stem}.npz"
        csv_path = out_dir / f"{stem}.csv"
        np.savez(
            npz_path,
            tau_c_grid=self.tc,
            tau_e_grid=self.te,
            delta_grid=self.dm,
            gains=self.gains,
            margins=self.margins,
        )
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tau_c_s", "tau_e_s", "delta_max_s", "kp", "kv", "ka", "dual_cert_margin", "certified"])
            for ic, tc_v in enumerate(self.tc):
                for ie, te_v in enumerate(self.te):
                    for id_, dm_v in enumerate(self.dm):
                        kp, kv, ka = self.gains[ic, ie, id_]
                        m = float(self.margins[ic, ie, id_])
                        writer.writerow([
                            f"{float(tc_v):.3f}",
                            f"{float(te_v):.3f}",
                            f"{float(dm_v):.3f}",
                            f"{float(kp):.4f}",
                            f"{float(kv):.4f}",
                            f"{float(ka):.4f}",
                            f"{m:.6f}",
                            str(m > 0.0),
                        ])
        return npz_path, csv_path

    @classmethod
    def load(cls, npz_path: Path) -> "CertifiedGainLUT":
        data = np.load(npz_path)
        return cls(
            tau_c_grid=data["tau_c_grid"],
            tau_e_grid=data["tau_e_grid"],
            delta_grid=data["delta_grid"],
            gains=data["gains"],
            margins=data["margins"],
        )

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


def synthesize_certified_lut(params: PlatoonParams = PlatoonParams()) -> CertifiedGainLUT:
    """Synthesize a real Dual-Certificate-verified 4 x 4 x 3 (48-cell) gain LUT.

    At each operating point (tau_c, tau_e, delta_max), evaluates candidate (kp, kv, ka)
    gains around the nominal operating envelope and selects the gain triple maximizing
    the exact Dual-Certificate margin subject to strict positivity.
    """
    tc_grid = np.linspace(0.05, 0.20, 4)
    te_grid = np.linspace(0.20, 0.50, 4)
    dm_grid = np.linspace(0.02, 0.08, 3)

    w = np.concatenate([np.linspace(params.omega_min, 2.5, 120), np.geomspace(2.5, params.omega_bar, 90)])
    kp_cands = np.array([1.05, 1.15, 1.25, 1.40, 1.55], dtype=np.float64)
    kv_cands = np.array([1.10, 1.18, 1.30, 1.45, 1.60], dtype=np.float64)
    ka_cands = np.array([0.75, 0.80, 0.85], dtype=np.float64)

    gains = np.zeros((4, 4, 3, 3), dtype=np.float32)
    margins = np.zeros((4, 4, 3), dtype=np.float32)

    for ic, tc_v in enumerate(tc_grid):
        for ie, te_v in enumerate(te_grid):
            for id_, dm_v in enumerate(dm_grid):
                best_m = -1e9
                best_g = (params.kp, params.kv, params.ka)
                for kp in kp_cands:
                    for kv in kv_cands:
                        for ka in ka_cands:
                            # Mild regularization toward nominal (1.15, 1.18, 0.80) for smooth scheduling
                            m = _dual_cert_margin(kp, kv, ka, params.headway, te_v, tc_v, dm_v, w)
                            if m > 0.0:
                                reg = 0.015 * ((kp - params.kp) ** 2 + (kv - params.kv) ** 2 + (ka - params.ka) ** 2)
                                score = m - reg
                                if score > best_m:
                                    best_m = score
                                    best_g = (kp, kv, ka)
                exact_m = _dual_cert_margin(
                    best_g[0], best_g[1], best_g[2], params.headway, te_v, tc_v, dm_v, w
                )
                gains[ic, ie, id_] = np.array(best_g, dtype=np.float32)
                margins[ic, ie, id_] = np.float32(exact_m)

    return CertifiedGainLUT(tc_grid, te_grid, dm_grid, gains, margins)


def benchmark_lut(
    n_queries: int = 20000,
    seed: int = 20260816,
    save_dir: Path | None = None,
    load_path: Path | None = None,
) -> dict:
    """Benchmark a real Dual-Certificate-verified 48-cell (4 x 4 x 3) gain LUT."""
    if load_path is not None and load_path.exists():
        lut = CertifiedGainLUT.load(load_path)
    else:
        lut = synthesize_certified_lut()
        if save_dir is not None:
            lut.save(save_dir)

    rng = np.random.default_rng(seed)
    q_tc = rng.uniform(0.05, 0.20, n_queries)
    q_te = rng.uniform(0.20, 0.50, n_queries)
    q_dm = rng.uniform(0.02, 0.08, n_queries)

    results = {
        "grid_points": int(lut.gains.size // 3),
        "certified_cells": lut.certified_cells,
        "min_dual_cert_margin": round(float(np.min(lut.margins)), 5),
        "memory_bytes": lut.memory_bytes,
        "memory_kb": round(lut.memory_bytes / 1024.0, 2),
        "seed": seed,
    }
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
