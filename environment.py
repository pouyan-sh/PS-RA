"""
environment.py
Gym-like OFDMA environment integrating channel models, JCM lookup and reward computation.
"""

import numpy as np
import math
from config import cfg
from utils import sample_rayleigh, sample_shadowing_db, compute_channel_gain, sinr_matrix, ber_qam_approx, JCMTable, lin2db, clamp, compute_delay
import random

class OFDMAEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.U = cfg.U
        self.B = cfg.B
        # load JCM tables
        self.jcm_class = JCMTable(cfg.jcm_class_csv, cfg.modulations)
        self.jcm_rec = JCMTable(cfg.jcm_rec_csv, cfg.modulations)
        self.reset_episode()

    def reset_episode(self):
        # per-episode channel params:
        self.distances_km = np.random.uniform(0.1, 2.0, size=(self.U, self.B))
        self.rayleigh = sample_rayleigh((self.U, self.B))
        self.shadow_db = sample_shadowing_db((self.U, self.B), self.cfg.shadowing_std_db)
        self.channel_gains = compute_channel_gain(self.rayleigh, self.shadow_db, self.distances_km)
        # user parameters
        self.tasks = np.random.choice(['classification', 'reconstruction'], size=self.U)
        self.alfas = np.zeros(self.U) # User-defined weight for balancing distortion vs. perception in their chosen task
        for i in range(self.U):
            # base = np.random.uniform(0.2, 0.8)
            base = np.random.uniform(0, 1)
            if self.tasks[i] == 'classification':
                # self.alfas[i] = max(0.0, base - 0.2)
                self.alfas[i] = min(base, 1- base )
            else:
                # self.alfas[i] = min(1.0, base + 0.2)
                self.alfas[i] = max(base, 1- base )
        self.num_images = np.random.randint(self.cfg.num_images_min, self.cfg.num_images_max + 1, size=self.U)
        self.BER_th = np.random.uniform(1e-4, 1e-2, size=self.U)
        self.sla = np.random.choice(cfg.service_levels, size=self.U)
        self.avg_weigth = np.random.uniform(0.0, 1.0, size=self.U)
        self.SSM = np.zeros(self.U)
        self.SSM_min = np.zeros(self.U)
        self.SSM_max = np.zeros(self.U)
        for i in range(self.U):
            a,b = np.random.uniform(0, 1,2)
            self.SSM_min[i] = min(a, b)
            self.SSM_max[i] = max(a, b)
        self.last_powers = np.zeros((self.U, self.B))
        self.delay_th = np.random.uniform(0.0, 1.0, size=self.U)
        self.delays = np.zeros(self.U)
        self.Phis = np.zeros(self.U)
        self.t = 0

    def reset(self):
        self.reset_episode()
        obs = [self._build_obs(i) for i in range(self.U)]
        return obs

    def _build_obs(self, i):
        return {
            'task': self.tasks[i],
            'delay': float(self.delays[i]),
            'alfa': float(self.alfas[i]), #
            'SSM': float(self.SSM[i]),
            'cr': float(np.random.uniform(self.cfg.CR_min, self.cfg.CR_max)),
            'P_max': float(self.cfg.P_max),
            'BER_th': float(self.BER_th[i]),
            'Phis': float(self.Phis[i]),
            'avg_weigth': float(self.avg_weigth[i]),
            'num_images': int(self.num_images[i]),
            'rb_gains': self.channel_gains[i, :].astype(float),
            'rb_interference': np.sum(self.last_powers * self.channel_gains, axis=0) - (self.last_powers[i, :] * self.channel_gains[i, :])
        }

    def step(self, actions_list):
        """
        actions_list: list of flattened action vectors (mask(B), power(B), cr(1), mod_logits(n_mod))
        """
        U = self.U; B = self.B
        masks = np.zeros((U, B))
        powers = np.zeros((U, B))
        crs = np.zeros(U)
        Ms = np.zeros(U, dtype=int)
        for i in range(U):
            a = actions_list[i]
            mask = a[:B]
            power = a[B:2*B]
            cr = np.clip(a[2*B], self.cfg.CR_min, self.cfg.CR_max)
            mod_logits = a[2*B+1:2*B+1+self.cfg.modulations.__len__()]
            # threshold mask
            mask_bin = (mask > 0.5).astype(float)
            if mask_bin.sum() == 0:
                mask_bin[np.argmax(mask)] = 1.0
            masks[i, :] = mask_bin
            # apply mask to power
            p = power * mask_bin
            total = p.sum()
            if total > 1e-12:
               if total > self.cfg.P_max:
                  p = p * (self.cfg.P_max / total) # ensure <= P_max

            powers[i, :] = p
            crs[i] = cr
            m_idx = int(np.argmax(mod_logits))
            Ms[i] = self.cfg.modulations[m_idx]
        # SINR
        sinr = sinr_matrix(powers, self.channel_gains, self.cfg.N0)
        # BER per RB
        ber = np.zeros_like(sinr)
        for i in range(U):
            for b in range(B):
                ber[i,b] = ber_qam_approx(sinr[i,b], Ms[i])
        # rates
        rates = np.sum(masks * self.cfg.W * np.log2(1 + sinr + 1e-12), axis=1)
        # delay
        delays = compute_delay(self.num_images, crs, rates, cfg.image_size)
        # SSM via JCM lookup
        SSMs = np.zeros(U)
        num_satisfy = 0 
        for i in range(U):
            sel = masks[i,:] > 0
            if sel.sum() == 0:
                eff_snr_db = -100
            else:
                eff_snr_linear = np.mean(sinr[i, sel])
                eff_snr_db = lin2db(eff_snr_linear + 1e-12)
            table = self.jcm_class if self.tasks[i] == 'classification' else self.jcm_rec
            D, PQ = table.lookup(eff_snr_db, Ms[i])

            lam = self.alfas[i]
            SSMs[i] = lam * PQ + (1 - lam) * (1 - D)
            if SSMs[i] > self.SSM_min[i]:
                num_satisfy += 1

        # local rewards
        R_max = (self.cfg.W * self.cfg.B)*math.log2(1.0 + (self.cfg.P_max/(self.cfg.B*self.cfg.N0)))
        local_rewards = np.zeros(U)
        Phis = np.zeros(U)
        for i in range(U):
            SSM_i = clamp((SSMs[i] - self.SSM_min[i]) / (self.SSM_max[i] - self.SSM_min[i]))
            sel = masks[i,:] > 0
            avg_ber = float(ber[i,sel].mean()) if sel.sum()>0 else 1.0
            f_ber = clamp(self.BER_th[i]/(avg_ber + 1e-12))
            f_delay = np.exp(- max(0, delays[i] - self.delay_th[i]) / self.delay_th[i])
            Phis[i] = (self.avg_weigth[i]/crs[i])*self.sla[i]
            local_rewards[i] = Phis[i]*(rates[i]/(R_max + 1e-12))*SSM_i*(self.cfg.theta1*SSM_i + self.cfg.theta2*f_ber + self.cfg.theta3*f_delay)
        
        # global reward
        collisions = np.clip(np.sum(masks, axis=0) - 1.0, 0.0, None)
        collisions_term = np.sum(collisions)
        sinr_term = 0.0
        active_rb = 0
        # for i in range(U):
        #     for b in range(B):
        #         if masks[i,b] > 0:
        #             sinr_term += masks[i,b] * math.log10(max(1e-12, sinr[i,b]))
        for b in range(B):
            if np.sum(masks[:, b]) > 0:   # RB فعال
               active_rb += 1
               for i in range(U):
                   if masks[i,b] > 0:
                      sinr_term += masks[i,b] * math.log10(max(sinr[i,b], 1e-12))
        if active_rb > 0:
           sinr_term /= active_rb
        else:
           sinr_term = 0.0
        global_reward = (self.cfg.lambda1 / U) * np.sum(local_rewards) - self.cfg.lambda2*(num_satisfy / U) - self.cfg.lambda3*collisions_term - self.cfg.lambda4 * sinr_term
        
        # update internal
        self.last_powers = powers
        self.SSM = SSMs
        self.t += 1
        next_obs = [self._build_obs(i) for i in range(U)]
        done = (self.t >= self.cfg.episode_length)
        info = {'SSM': SSMs, 'rates': rates}
        return next_obs, local_rewards, global_reward, done, info
