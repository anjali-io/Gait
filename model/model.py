import math
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

# --------- Network imports ---------
from .network.triplet import TripletLoss
from .network.cvl_model import CVL_GaitSet
from .utils import TripletSampler


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=64,
                 vtm_hidden=512):

        self.save_name = save_name
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.img_size = img_size

        # --------- Encoder ---------
        self.encoder = nn.DataParallel(
            CVL_GaitSet(hidden_dim=hidden_dim, vtm_hidden=vtm_hidden).float()
        ).cuda()

        # --------- Triplet Loss ---------
        self.triplet_loss = nn.DataParallel(
            TripletLoss(self.P * self.M, hard_or_full_trip, margin).float()
        ).cuda()

        # --------- Optimizer ---------
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)

        self.sample_type = 'all'

    # -------------------------------------------------
    # Collate function (UNCHANGED)
    # -------------------------------------------------
    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])

        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]

        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)])
                    for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)

            batch_frames = [[
                len(frame_sets[i])
                for i in range(batch_per_gpu * g, batch_per_gpu * (g + 1))
                if i < batch_size
            ] for g in range(gpu_num)]

            for g in range(len(batch_frames)):
                while len(batch_frames[g]) < batch_per_gpu:
                    batch_frames[g].append(0)

            max_sum_frame = max(sum(bf) for bf in batch_frames)

            seqs = [[
                np.concatenate([
                    seqs[i][j]
                    for i in range(batch_per_gpu * g, batch_per_gpu * (g + 1))
                    if i < batch_size
                ], 0)
                for g in range(gpu_num)]
                for j in range(feature_num)
            ]

            seqs = [np.asarray([
                np.pad(seqs[j][g],
                       ((0, max_sum_frame - seqs[j][g].shape[0]), (0, 0), (0, 0)),
                       'constant')
                for g in range(gpu_num)])
                for j in range(feature_num)
            ]

            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    # -------------------------------------------------
    # Training loop (ITERATION-BASED ✅)
    # -------------------------------------------------
    def fit(self):
        self.encoder.train()
        self.sample_type = 'random'

        sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

        train_label_set = sorted(self.train_source.label_set)

        while self.restore_iter < self.total_iter:
            for seq, view, seq_type, label, batch_frame in train_loader:

                self.restore_iter += 1
                self.optimizer.zero_grad()

                for i in range(len(seq)):
                    seq[i] = self.np2var(seq[i]).float()

                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()

                out = self.encoder(seq[0], batch_frame)
                feature = out['feature'] if isinstance(out, dict) else out[0]

                # 🔥 L2 NORMALIZATION (VERY IMPORTANT)
                feature = feature / (feature.norm(p=2, dim=-1, keepdim=True) + 1e-12)

                target_label = [train_label_set.index(l) for l in label]
                target_label = self.np2var(np.array(target_label)).long()

                triplet_feature = feature.permute(1, 0, 2).contiguous()
                triplet_label = target_label.unsqueeze(0).repeat(
                    triplet_feature.size(0), 1
                )

                full_loss, hard_loss, _, _ = self.triplet_loss(
                    triplet_feature, triplet_label
                )

                loss = hard_loss.mean() if self.hard_or_full_trip == 'hard' else full_loss.mean()
                loss.backward()
                self.optimizer.step()

                if self.restore_iter % 500 == 0:
                    print(f"[Iter {self.restore_iter}/{self.total_iter}] loss={loss.item():.4f}")
                    sys.stdout.flush()

                if self.restore_iter >= self.total_iter:
                    break

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------
    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    # -------------------------------------------------
    # Feature extraction (WITH L2 NORMALIZATION ✅)
    # -------------------------------------------------
    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'

        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

        feature_list, view_list, seq_type_list, label_list = [], [], [], []

        with torch.no_grad():
            for seq, view, seq_type, label, batch_frame in data_loader:
                for i in range(len(seq)):
                    seq[i] = self.np2var(seq[i]).float()

                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()

                out = self.encoder(seq[0], batch_frame)
                feature = out['feature'] if isinstance(out, dict) else out[0]

                # 🔥 L2 NORMALIZATION
                feature = feature / (feature.norm(p=2, dim=-1, keepdim=True) + 1e-12)

                n, _, _ = feature.size()
                feature_list.append(feature.view(n, -1).cpu().numpy())

                view_list += view
                seq_type_list += seq_type
                label_list += label

        return (
            np.concatenate(feature_list, 0),
            view_list,
            seq_type_list,
            label_list
        )




