import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y, device=None):
    """
    Compute pairwise Euclidean distance.

    Args:
        x: (N, D) numpy array
        y: (M, D) numpy array
    Returns:
        dist: (N, M) torch tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    with torch.no_grad():
        dist = (
            torch.sum(x ** 2, dim=1, keepdim=True)
            + torch.sum(y ** 2, dim=1, keepdim=True).transpose(0, 1)
            - 2.0 * torch.matmul(x, y.transpose(0, 1))
        )
        dist = torch.sqrt(F.relu(dist))

    return dist


def evaluation(data, config):
    """
    CASIA-B evaluation protocol (Rank-1 ~ Rank-5).

    Args:
        data: (feature, view, seq_type, label)
            feature : (N, D) numpy array
            view     : list of view labels ('000', '018', ..., '180')
            seq_type : list of sequence types (nm-01, bg-01, cl-01)
            label    : list of identity labels
        config: dict with key 'dataset' (CASIA-B)

    Returns:
        acc: numpy array with shape
             [3, num_probe_views, num_gallery_views, 5]
             probe types = [NM, BG, CL]
    """

    # ---- Dataset (CASIA-B assumed) ----
    dataset = config['dataset']   # 'CASIA-B'

    feature, view, seq_type, label = data
    label = np.asarray(label)

    view_list = sorted(list(set(view)))
    view_num = len(view_list)

    # ---- CASIA-B protocol ----
    probe_seq_dict = [
        ['nm-05', 'nm-06'],   # Normal walking
        ['bg-01', 'bg-02'],   # Bag
        ['cl-01', 'cl-02']    # Coat
    ]

    gallery_seq = ['nm-01', 'nm-02', 'nm-03', 'nm-04']

    num_rank = 5
    acc = np.zeros((3, view_num, view_num, num_rank), dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for p, probe_seq in enumerate(probe_seq_dict):
        for v1, probe_view in enumerate(view_list):
            for v2, gallery_view in enumerate(view_list):

                # ---- Gallery samples ----
                gallery_mask = (
                    np.isin(seq_type, gallery_seq)
                    & np.isin(view, [gallery_view])
                )
                gallery_x = feature[gallery_mask]
                gallery_y = label[gallery_mask]

                # ---- Probe samples ----
                probe_mask = (
                    np.isin(seq_type, probe_seq)
                    & np.isin(view, [probe_view])
                )
                probe_x = feature[probe_mask]
                probe_y = label[probe_mask]

                if len(probe_x) == 0 or len(gallery_x) == 0:
                    continue

                # ---- Distance computation ----
                dist = cuda_dist(probe_x, gallery_x, device)
                idx = torch.argsort(dist, dim=1).cpu().numpy()

                # ---- Rank-k accuracy ----
                for r in range(num_rank):
                    correct = (
                        probe_y.reshape(-1, 1)
                        == gallery_y[idx[:, :r + 1]]
                    )
                    acc[p, v1, v2, r] = np.round(
                        np.mean(np.any(correct, axis=1)) * 100.0, 2
                    )

    return acc


