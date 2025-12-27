"""
utils.py
Utilities: pathloss, conversions, channel sampling, BER approx and JCM table loader.
"""

import numpy as np
import pandas as pd
import math
from typing import List, Tuple
import ast


def sample_user_positions(num_users, cell_radius_km, bs_pos=(0.0, 0.0)):
    """
    Uniform user distribution inside a circular cell.

    Returns:
        distances_km: (U,)
        positions: (U,2)
    """
    r = cell_radius_km * np.sqrt(np.random.uniform(0, 1, num_users))
    theta = np.random.uniform(0, 2*np.pi, num_users)

    x = r * np.cos(theta) + bs_pos[0]
    y = r * np.sin(theta) + bs_pos[1]

    dx = x - bs_pos[0]
    dy = y - bs_pos[1]

    distances_km = np.sqrt(dx**2 + dy**2)

    return distances_km, np.stack([x, y], axis=1)



def pathloss_db(distance_km: float) -> float:
    """
    Pathloss (dB): PL(dB) = 128.1 + 37.6 * log10(d_km)
    """
    if distance_km <= 0:
        distance_km = 1e-3
    return 128.1 + 37.6 * math.log10(distance_km)

def db2lin(db: float) -> float:
    return 10 ** (db / 10.0)

def lin2db(x: float) -> float:
    return 10 * math.log10(max(x, 1e-30))

def sample_rayleigh(size):
    """|h|^2 ~ exponential(1)"""
    return np.random.exponential(scale=1.0, size=size)

def sample_shadowing_db(size, sigma_db):
    return np.random.normal(0.0, sigma_db, size=size)

def compute_channel_gain(rayleigh_mag2: np.ndarray, shadow_db: np.ndarray, dist_km: np.ndarray) -> np.ndarray:
    """
    Linear power gain = |h|^2 * 10^(shadow_db/10) * 10^(-PL(dB)/10)
    All inputs arrays must be same shape.
    """
    pl_db = np.vectorize(pathloss_db)(dist_km)
    shadow_lin = 10 ** (shadow_db / 10.0)
    pl_lin = 10 ** (-pl_db / 10.0)
    gains = rayleigh_mag2 * shadow_lin * pl_lin
    # print("Distances (km):", dist_km)
    # print("Pathloss (dB):", pl_db)

    return np.maximum(gains, 1e-15)

def sinr_matrix(powers: np.ndarray, gains: np.ndarray, N0: float) -> np.ndarray:
    """
    Compute SINR_{i,b} with interference.
    powers: (U,B), gains: (U,B)
    """
    
    signal = powers * gains  # (U,B)
    # total received power per RB
    total_rb = np.sum(signal, axis=0, keepdims=True)  # (1,B)
    # interference for each user is total minus own signal
    interference = total_rb - signal  # (U,B)
    interference = np.maximum(interference, 0.0)
    denom = N0 + interference
    sinr = np.divide(signal, denom, out=np.zeros_like(signal), where=(denom > 0))
    # print('---------------powers---------------')
    # print(powers)
    # print('---------------gains---------------')
    # print(gains)
    # print('---------------signal---------------')
    # print(signal)
    # print('---------------total_rb---------------')
    # print(total_rb)
    # print('---------------interference---------------')
    # print(interference)
    # print('---------------N0---------------')
    # print(N0)
    # print('---------------denom---------------')
    # print(denom)
    # print('---------------sinr---------------')
    # print(sinr)

    return sinr

def ber_qam_approx(snr_linear: float, M: int) -> float:
    """
    Approximate BER for square M-QAM using common approximation.
    Return value in [0,1].
    """
    if snr_linear <= 0:
        return 0.5
    k = math.log2(M)
    factor = math.sqrt((3.0 * k) / (M - 1.0) * snr_linear)
    # Q(x) approx
    q = 0.5 * math.erfc(factor / math.sqrt(2.0))
    ber = (4.0 / k) * (1 - 1.0 / math.sqrt(M)) * q
    return float(min(max(ber, 0.0), 1.0))

def clamp(x, min_val=0.0, max_val=1.0):
    return max(min_val, min(x, max_val))

def compute_delay(num_images, comp_ratio, rates, image_size, delay_cap=1.0):
    data_bits = num_images * image_size
    compressed = data_bits * comp_ratio
    delay = np.where(rates > 1e-9, compressed / rates, delay_cap)
    return np.minimum(delay, delay_cap)


def compression_to_features(cr, total_features, min_features, max_features, step):
    """
    Convert compression ratio to number of features.
    """

    # features after compression
    raw_feat = int(round(total_features * cr))

    # clip to valid range
    raw_feat = np.clip(raw_feat, min_features, max_features)

    # snap to nearest valid step (e.g., 76,78,...)
    snapped = min_features + step * round((raw_feat - min_features) / step)

    return int(np.clip(snapped, min_features, max_features))



class JCMTable:
    """
    CSV format:
        snr_db, M4, M16, M64 ...
    where each M column contains a 2-value object such as:
        [D, PQ]
        or "D PQ"
        or "D,PQ"
    """

    def __init__(self, csv_path, mod_list, D_idx, PQ_idx, snr_col=None):
        self.df = pd.read_csv(csv_path)
        self.mod_list = mod_list
        self.D_idx = D_idx
        self.PQ_idx = PQ_idx

        # --- SNR column handling ---
        if snr_col is None:
           # assume first column is SNR
           self.snr_vals = self.df.iloc[:, 0].to_numpy()
           self.snr_vals = self.snr_vals.astype(float)
           self.mod_columns = self.df.columns[1:]
        else:
           if snr_col not in self.df.columns:
              raise ValueError(f"CSV must contain '{snr_col}' column.")
           self.snr_vals = self.df[snr_col].to_numpy()
           self.mod_columns = [c for c in self.df.columns if c != snr_col]

        # sanity check modulation columns
        for M in mod_list:
           col = f"{M}QAM" if f"{M}QAM" in self.mod_columns else f"M{M}"
           if col not in self.df.columns:
              raise ValueError(f"Missing modulation column: {col}")

    def _parse_cell(self, cell):
        """
        Parses a cell containing multiple values and extracts D and PQ.
        """

        if isinstance(cell, (list, tuple, np.ndarray)):
           vals = cell
        elif isinstance(cell, str):
           cell = cell.strip()
           if cell.startswith("["):
              vals = ast.literal_eval(cell)
           else:
            vals = [float(x) for x in cell.replace(",", " ").split()]
        else:
            raise ValueError(f"Unsupported cell type {type(cell)}")

        D = float(vals[self.D_idx])
        PQ = float(vals[self.PQ_idx])

        return D, PQ

    def lookup(self, snr_db, M):
        """
        Returns (D, PQ) for modulation M at nearest SNR.
        """
        if f"{M}QAM" in self.df.columns:
           col = f"{M}QAM"
        else:
           col = f"M{M}"
        idx = np.argmin(np.abs(self.snr_vals - snr_db))

        cell = self.df.iloc[idx][col]
        D, PQ = self._parse_cell(cell)
        return D, PQ
