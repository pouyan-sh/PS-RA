"""
utils.py
Utilities: pathloss, conversions, channel sampling, BER approx and JCM table loader.
"""

import numpy as np
import pandas as pd
import math
from typing import List, Tuple
import ast

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
    return rayleigh_mag2 * shadow_lin * pl_lin

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
    denom = N0 + interference
    sinr = np.divide(signal, denom, out=np.zeros_like(signal), where=(denom > 0))
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

def compute_delay(num_images, comp_ratio, rates, image_size):
    data_bits = num_images * image_size
    compressed = data_bits * comp_ratio
    delay = np.where(rates > 0, compressed / rates, 1000.0)
    return delay



class JCMTable:
    """
    CSV format:
        snr_db, M4, M16, M64 ...
    where each M column contains a 2-value object such as:
        [D, PQ]
        or "D PQ"
        or "D,PQ"
    """

    def __init__(self, csv_path, mod_list):
        self.df = pd.read_csv(csv_path)
        self.mod_list = mod_list

        if "snr_db" not in self.df.columns:
            raise ValueError("CSV must contain 'snr_db' column.")

        self.snr_vals = self.df["snr_db"].to_numpy()

        # validate existence of columns like M4, M16, ...
        for M in mod_list:
            col = f"M{M}"
            if col not in self.df.columns:
                raise ValueError(f"CSV missing modulation column: {col}")

    def _parse_pair(self, cell):
        """
        Converts cell content into (D, PQ).
        Handles formats:
            [0.9, 0.1]
            "[0.9,0.1]"
            "0.9 0.1"
            "0.9,0.1"
        """
        if isinstance(cell, (list, tuple, np.ndarray)):
            if len(cell) != 2:
                raise ValueError("Mod cell must contain 2 values: [D, PQ]")
            return float(cell[0]), float(cell[1])

        # If string, try to parse
        if isinstance(cell, str):
            cell = cell.strip()

            # Case like "[0.9,0.1]"
            if cell.startswith("[") and cell.endswith("]"):
                arr = ast.literal_eval(cell)
                return float(arr[0]), float(arr[1])

            # Case like "0.9 0.1" or "0.9,0.1"
            if "," in cell:
                parts = cell.split(",")
            else:
                parts = cell.split()

            if len(parts) != 2:
                raise ValueError(f"Cannot parse mod cell '{cell}', expected 2 values.")

            return float(parts[0]), float(parts[1])

        raise ValueError(f"Unsupported cell type: {type(cell)}")

    def lookup(self, snr_db, M):
        """
        Returns (D, PQ) for modulation M at nearest SNR.
        """
        col = f"M{M}"
        idx = np.argmin(np.abs(self.snr_vals - snr_db))

        cell = self.df.iloc[idx][col]
        D, PQ = self._parse_pair(cell)
        return D, PQ
