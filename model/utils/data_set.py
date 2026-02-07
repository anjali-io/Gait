import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import cv2
import xarray as xr


class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        self.view = view              # view string like '090'
        self.seq_type = seq_type
        self.label = label            # raw labels (strings)
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution) / 64 * 10)

        # ---- Dataset size ----
        self.data_size = len(self.label)

        # ---- Caches ----
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        # ---- CRITICAL FIX: identity set MUST be INT ----
        self.label_set = sorted(list(set(int(l) for l in self.label)))

    def __len__(self):
        return self.data_size

    def __loader__(self, path):
        """
        Load one gait sequence directory and return normalized silhouettes.
        """
        arr = self.img2xarray(path)[:, :, self.cut_padding:-self.cut_padding]
        return arr.astype("float32") / 255.0

    def __getitem__(self, index):
        # ---- Load sequence ----
        if not self.cache:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
        elif self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            self.data[index] = data
        else:
            data = self.data[index]

        # ---- Frame indices ----
        if self.frame_set[index] is None:
            self.frame_set[index] = list(range(data[0].shape[0]))

        seq = data
        frame_set = self.frame_set[index]

        return (
            seq,                   # sequence data
            frame_set,             # frame indices
            self.view[index],      # view string ('000', '090', ...)
            self.seq_type[index],  # sequence type (nm-01, bg-01, cl-01)
            int(self.label[index]) # identity label (INT)
        )

    def img2xarray(self, file_path):
        """
        Read silhouette images from a directory and return xarray DataArray.
        """
        imgs = sorted(os.listdir(file_path))
        frames = [
            cv2.imread(osp.join(file_path, img_path), cv2.IMREAD_GRAYSCALE)
            for img_path in imgs
            if osp.isfile(osp.join(file_path, img_path))
        ]

        frames = [
            np.reshape(img, (self.resolution, self.resolution))
            for img in frames
        ]

        return xr.DataArray(
            frames,
            coords={'frame': list(range(len(frames)))},
            dims=['frame', 'img_y', 'img_x'],
        )



