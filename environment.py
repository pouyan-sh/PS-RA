"""
Gym-like OFDMA environment integrating channel models, JCM lookup and reward computation.
"""

import numpy as np
import math
from config import cfg
from utils import sample_rayleigh, sample_shadowing_db, compute_channel_gain, sinr_matrix, ber_qam_approx, JCMTable, lin2db, clamp, compute_delay, sample_user_positions, compression_to_features
import random

class OFDMAEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.U = cfg.U
        self.B = cfg.B
        
        self.jcm_tables = {}  # feature_count -> JCMTable
        # load JCM tables
        

        self.last_masks = np.zeros((self.U, self.B))
        self.last_sinr = np.zeros((self.U, self.B))
        self.last_rates = np.zeros(self.U)

        self.base_shadow_db = None  # Base per episode


        self.reset_episode()

    def reset_episode(self):
        # per-episode channel params:
        self.distances_km, self.user_positions = sample_user_positions( self.U, self.cfg.cell_radius_km, self.cfg.bs_position)
        self.rayleigh = sample_rayleigh((self.U, self.B))
        # self.shadow_db = sample_shadowing_db((self.U, self.B), self.cfg.shadowing_std_db)
        self.shadow_db = sample_shadowing_db(self.U, self.cfg.shadowing_std_db)
        dist_mat = self.distances_km[:, None]     # (U,1)
        shadow_mat = self.shadow_db[:, None]       # (U,1)

        # self.channel_gains = compute_channel_gain(self.rayleigh, self.shadow_db, self.distances_km)
        self.channel_gains = compute_channel_gain(self.rayleigh, shadow_mat, dist_mat)


        # print("BS position:", self.cfg.bs_position)
        # print("User positions (x,y):", self.user_positions)
        # print("User distances (km):", self.distances_km)
        # print("Pathloss (dB):",  128.1 + 37.6 * np.log10(self.distances_km))


        # user parameters
        self.tasks = np.random.choice(['classification', 'reconstruction'], size=self.U)
        self.alfas = np.zeros(self.U) # User-defined weight for balancing distortion vs. perception in their chosen task
        for i in range(self.U):
            # base = np.random.uniform(0.2, 0.8)
            base = np.round(np.random.uniform(0, 1) , 3)
            if self.tasks[i] == 'classification':
                # self.alfas[i] = max(0.0, base - 0.2)
                self.alfas[i] = min(base, 1- base )
            else:
                # self.alfas[i] = min(1.0, base + 0.2)
                self.alfas[i] = max(base, 1- base )
        self.num_images = np.random.randint(self.cfg.num_images_min, self.cfg.num_images_max + 1, size=self.U)
        self.BER_th = np.round(np.random.uniform(1e-5, 1e-3, size=self.U),3)
        self.sla = np.random.choice(cfg.service_levels, size=self.U)
        self.avg_weigth =  np.round(np.random.uniform(0.5, 1.0, size=self.U) , 3)
        self.SSM = np.zeros(self.U)
        self.SSM_min = np.zeros(self.U)
        self.SSM_max = np.zeros(self.U)
        self.delay_violation = np.zeros(self.U)
        self.max_violation = 100

        for i in range(self.U):
            ssm_min = np.round(np.random.uniform(0.0, 0.6), 3)
            ssm_max = np.round(np.random.uniform(ssm_min + cfg.min_gap, 1.0), 3)
            self.SSM_min[i] = ssm_min
            self.SSM_max[i] = ssm_max

        self.last_powers = np.zeros((self.U, self.B))
        self.delay_th = np.round(np.random.uniform(cfg.delay_th_min, cfg.delay_th_max, size=self.U) , 3)
        self.delays = np.zeros(self.U)
        self.Phis = np.zeros(self.U)
        self.last_masks = np.zeros((self.U, self.B))
        self.last_rates = np.zeros(self.U)
        self.last_sinr = np.zeros((self.U, self.B))
        self.total_power = np.zeros(self.U)

        self.t = 0

    def reset(self):
        self.reset_episode()
        obs = [self._build_obs(i) for i in range(self.U)]
        return obs

    def _build_obs(self, i):
        return {
            'task': 0.0 if self.tasks[i] == 'classification' else 1.0,
            'delay': float(self.delays[i]),
            'alfa': float(self.alfas[i]), 
            'SSM': float(self.SSM[i]),
            'cr': float(np.random.uniform(self.cfg.CR_min, self.cfg.CR_max)),
            'P_max': float(self.cfg.P_max),
            'BER_th': float(self.BER_th[i]),
            'Phis': float(self.Phis[i]),
            'avg_weigth': float(self.avg_weigth[i]),
            'num_images': int(self.num_images[i]),
            'rb_gains': self.channel_gains[i, :].astype(float),
            'delay_violation': float(self.delay_violation[i] / self.max_violation),
            'rb_interference': np.maximum(np.sum(self.last_powers * self.channel_gains, axis=0) - (self.last_powers[i, :] * self.channel_gains[i, :]), 0.0),
            'rb_mask_prev': self.last_masks[i, :],
            'rb_power_prev': self.last_powers[i, :],
            'sinr_prev': self.last_sinr[i, :],
            'rate_prev': float(self.last_rates[i]),
            'num_active_rb': int(np.sum(self.last_masks[i])),
            'rb_load': np.sum(self.last_masks, axis=0),
            'num_collided_rb': int(np.sum(np.sum(self.last_masks, axis=0) > 1)),
            'power_usage': float(np.clip(self.total_power[i] / self.cfg.P_max, 0.0, 2.0))

        }
    def get_jcm_table(self, feature_count):
        if feature_count not in self.jcm_tables:
            path = f"{self.cfg.jcm_table_dir}/{self.cfg.jcm_table_prefix}{feature_count}.csv"
            self.jcm_tables[feature_count] = JCMTable(
                path,
                self.cfg.modulations,
                self.cfg.jcm_D_index,
                self.cfg.jcm_PQ_index,
                self.cfg.jcm_snr_column
            )
        return self.jcm_tables[feature_count]

    def step(self, actions_list):
        """
        actions_list: list of flattened action vectors (mask(B), power(B), cr(1), mod_logits(n_mod))
        """

        # fast fading update (every step)
        self.rayleigh = sample_rayleigh((self.U, self.B))

        self.channel_gains = compute_channel_gain(self.rayleigh, self.shadow_db[:, None], self.distances_km[:, None])

        # slow shadowing update
        # rho = 0.99  # high temporal correlation
        # noise = np.random.normal(0.0, self.cfg.shadowing_std_db, size=(self.U, self.B))
        # self.shadow_db = rho * self.shadow_db + np.sqrt(1 - rho**2) * noise
        # self.shadow_db = np.clip(self.shadow_db, -20, 20)
        # self.channel_gains = compute_channel_gain(self.rayleigh, self.shadow_db, self.distances_km)

        # print("Shadowing dB stats:", np.min(self.shadow_db), np.mean(self.shadow_db), np.max(self.shadow_db))
        # print("N0 =", self.cfg.N0)

        self.total_power = np.zeros(self.U)
        U = self.U; B = self.B
        masks = np.zeros((U, B))
        powers = np.zeros((U, B))
        crs = np.zeros(U)
        Ms = np.zeros((U, B), dtype=int)
        power_violations = np.zeros(U)
        for i in range(U):
            a = actions_list[i]
            mask = a[:B]
            power = a[B:2*B]
            cr = np.clip(a[2*B], self.cfg.CR_min, self.cfg.CR_max)
            mod_start = 2*B + 1
            n_mod = len(self.cfg.modulations)

            mod_logits = a[mod_start : mod_start + B * n_mod]
            mod_logits = mod_logits.reshape(B, n_mod)   # (B, n_mod)

            # threshold mask
            mask_bin = (mask > 0.5).astype(float)
            if mask_bin.sum() == 0:
                mask_bin[np.argmax(mask)] = 1.0
            masks[i, :] = mask_bin
            # apply mask to power
            p = power * mask_bin
            powers[i, :] = p
            self.total_power [i] = np.sum(p)
            power_violations[i] = max(0.0, self.total_power[i] - self.cfg.P_max)
            crs[i] = cr
            for b in range(B):
               if mask_bin[b] > 0:
                  m_idx = int(np.argmax(mod_logits[b]))
                  Ms[i, b] = self.cfg.modulations[m_idx]
               else:
                   Ms[i, b] = 0

        # SINR
        sinr = sinr_matrix(powers, self.channel_gains, self.cfg.N0)
        sinr = np.maximum(sinr, 0.0)
        sinr_db = 10 * np.log10(sinr + 1e-12)

        # BER per RB
        ber = np.zeros_like(sinr)
        for i in range(U):
            for b in range(B):
                if Ms[i, b] > 0:
                   ber[i, b] = ber_qam_approx(sinr[i, b], Ms[i, b])
                else:
                   ber[i, b] = 1.0

        # rates
        rates = np.sum(masks * self.cfg.W * np.log2(1 + sinr + 1e-12), axis=1)
        # print('--------------------------------nasnasy------------------------------')
        # print (sinr)

        # delay
        delays = compute_delay(self.num_images, crs, rates, cfg.image_size, cfg.delay_cap)
        for i in range(self.U):
            if delays[i] > self.delay_th[i]:
               self.delay_violation[i] += 1
            self.delay_violation[i] = min(self.delay_violation[i], self.max_violation)
        # SSM via JCM lookup
        SSMs = np.zeros(U)
        num_satisfy = 0 
        for i in range(U):
            sel = masks[i,:] > 0
            if sel.sum() == 0:
               SSMs[i] = 0.0  # Low SSM if no RB
            else:
                rb_indices = np.where(sel)[0]
                ssm_per_rb = np.zeros(len(rb_indices))
                for b_idx, b in enumerate(rb_indices):
                    eff_snr_linear = sinr[i, b]  # Per RB SNR
                    eff_snr_db = lin2db(eff_snr_linear + 1e-12)
                    
                    feature_count = compression_to_features(
                       crs[i],
                       self.cfg.total_features,
                       self.cfg.min_features,
                       self.cfg.total_features,
                       self.cfg.feature_step
                    )
                    table = self.get_jcm_table(feature_count)
                    # clip SNR to table range
                    snr_min = float(table.snr_vals.min())
                    snr_max = float(table.snr_vals.max())
                    snr_clip = np.clip(
                        eff_snr_db,
                        table.snr_vals.min(),
                        table.snr_vals.max()
                    )

                    M = Ms[i, b]  # Per RB M
                    D, PQ = table.lookup(snr_clip, M)

                    lam = self.alfas[i]
                    ssm_per_rb[b_idx] = lam * PQ + (1 - lam) * (1 - D)
                # SSMs[i] = np.mean(ssm_per_rb)  # Average over RBs
                
                # rate per active RB (using already computed SINR)
                rb_rates = self.cfg.W * np.log2(1 + sinr[i, rb_indices] + 1e-12)
                sum_rate = np.sum(rb_rates)
                if sum_rate > 0:
                   weights = rb_rates / (sum_rate + 1e-12)  
                   SSMs[i] = np.sum(weights * ssm_per_rb)
                else:
                   SSMs[i] = 0.0


        # local rewards
        R_max = (self.cfg.W * self.cfg.B)*math.log2(1.0 + (self.cfg.P_max/(self.cfg.B*self.cfg.N0)))
        local_rewards = np.zeros(U)
        Phis = np.zeros(U)
        EE = np.zeros(U)
        for i in range(U):
            power_penalty = (power_violations[i] / self.cfg.P_max)
            EE[i] = (delays[i] * self.total_power [i])/self.cfg.P_max
            SSM_i = clamp((SSMs[i] - self.SSM_min[i]) / (self.SSM_max[i] - self.SSM_min[i]))
            sel = masks[i,:] > 0
            avg_ber = float(ber[i,sel].mean()) if sel.sum()>0 else 1.0
            f_ber = clamp(self.BER_th[i]/(avg_ber + 1e-12))
            den = max(self.delay_th[i], 1e-3)
            f_delay = np.exp(- max(0, delays[i] - self.delay_th[i]) / den)
            Phis[i] = (self.avg_weigth[i]/(crs[i] + self.cfg.eps_cr))*(self.sla[i] / self.cfg.max_sla)
            satisfy_bonus = 0.0
            if SSMs[i] > self.SSM_min[i] and avg_ber <= self.BER_th[i]:  
               satisfy_bonus = self.cfg.theta_satisfy

            local_rewards[i] = Phis[i]*(rates[i]/(R_max + 1e-12))*SSM_i*(self.cfg.theta1*SSM_i + self.cfg.theta2*f_ber + self.cfg.theta3*f_delay) + satisfy_bonus - (self.cfg.lambda_power * power_penalty) + ((rates[i]*SSM_i)/((EE[i]+ 1e-12)*(R_max + 1e-12)))
            
            if SSMs[i] >= self.SSM_min[i] and avg_ber <= self.BER_th[i]:
              num_satisfy += 1
        
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
            if np.sum(masks[:, b]) > 0:   
               active_rb += 1
               for i in range(U):
                   if masks[i,b] > 0:
                      sinr_term += masks[i,b] * math.log10(max(sinr[i,b], 1e-12))
        if active_rb > 0:
           sinr_term /= active_rb
        else:
           sinr_term = 0.0
        global_reward = (self.cfg.lambda1 / U) * np.sum(local_rewards) - self.cfg.lambda2 * ((U - num_satisfy) /  U) - self.cfg.lambda3*(collisions_term /B)#- self.cfg.lambda4 * sinr_term
        
        rb_user_count = np.sum(masks, axis=0)
        num_collided_rb = int(np.sum(rb_user_count > 1))
        self.last_powers = powers
        self.SSM = SSMs
        self.t += 1
        done = (self.t >= self.cfg.episode_length)
        avg_delay = float(np.mean(delays))
        satisfy_fraction = float(num_satisfy / U)

        info = {
             'SSM': SSMs,                      # per-user
             'rates': rates,                  # per-user
             'sinr_db':sinr_db,
             'delays': delays,                # per-user
             'avg_delay': avg_delay,           # scalar
             'satisfy_fraction': satisfy_fraction,  # scalar
             'num_collided_rb': num_collided_rb,
             'rb_user_count': rb_user_count.copy()
        }

        self.last_rates = rates
        self.last_sinr = sinr
        self.last_masks = masks.copy()
        next_obs = [self._build_obs(i) for i in range(U)]

        return next_obs, local_rewards, global_reward, done, info
