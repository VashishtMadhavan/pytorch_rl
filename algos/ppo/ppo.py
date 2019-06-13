import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from algos.a2c.a2c import A2C

class PPO(A2C):
    def _init_train_ops(self):
        super(PPO, self)._init_train_ops()
        # PPO specific parameters
        self.surr_batches = self.args.surr_batches
        self.surr_epochs = self.args.surr_epochs
        if self.threads >= self.surr_batches:
            self.surr_batch_size = self.threads // self.surr_batches
        else:
            self.surr_batch_size = 1
        self.clip_frac = self.args.clip_frac

    def loss(self, sample_dict, idx):
        forward_start_time = time.time()
        log_ratios, advantages, values, entropy = self._forward_policy(sample_dict, idx=idx, ratio=True)
        old_values = sample_dict['old_values'][:, idx]; returns = sample_dict['returns'][:, idx]
        update_start_time = time.time()

        # clipped pg loss
        ratio = torch.exp(log_ratios)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, min=1.0 - self.clip_frac, max=1.0 + self.clip_frac)
        pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

        # clipped value loss
        values_clipped = old_values + torch.clamp(values.squeeze() - old_values, min=-self.clip_frac, max=self.clip_frac)
        vf_loss1 = (values.squeeze() - returns).pow(2)
        vf_loss2 = (values_clipped - returns).pow(2)
        value_loss = torch.mean(torch.max(vf_loss1, vf_loss2))

        # entropy loss
        entropy_loss = torch.mean(entropy)
        self.logger.set("policy_loss", pg_loss.item())
        self.logger.set("value_loss", value_loss.item())
        self.logger.set("entropy", entropy_loss.item())
        return pg_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

    def update(self, sample_dict):
        avg_surr_epoch_time = []
        for _ in range(self.surr_epochs):
            for i in range(self.surr_batches):
                sample_idx = np.random.choice(self.threads, self.surr_batch_size, replace=False)
                self.optimizer.zero_grad()
                loss = self.loss(sample_dict, idx=sample_idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == '__main__':
    main()