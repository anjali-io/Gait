# -*- coding: utf-8 -*-
# @Author  : Abner (Modified by ChatGPT for Himanshu)
# @Time    : 2018/12/19

import os
import imageio
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

# ----------------------------
# IMPORT SETTINGS FROM config.py
# ----------------------------
from config import conf
DEFAULT_INPUT = conf["data"]["dataset_path"]                     # CASIA-B raw videos
DEFAULT_OUTPUT = "./CASIA-B-64"                                  # Processed silhouettes

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


# ----------------------------
# ARGUMENT PARSER (FIXED DEFAULTS)
# ----------------------------
parser = argparse.ArgumentParser(description='CASIA-B Pretreatment')

parser.add_argument('--input_path', default=DEFAULT_INPUT, type=str,
                    help='Root path of raw CASIA-B dataset.')

parser.add_argument('--output_path', default=DEFAULT_OUTPUT, type=str,
                    help='Path for processed silhouette output.')

parser.add_argument('--log_file', default='./pretreatment.log', type=str,
                    help='Log file path.')

parser.add_argument('--log', default=False, type=boolean_string,
                    help='Save all logs if True.')

parser.add_argument('--worker_num', default=1, type=int,
                    help='Number of worker processes.')

opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path
LOG_PATH = opt.log_file
IF_LOG = opt.log
WORKERS = opt.worker_num

T_H = 64
T_W = 64


# ----------------------------
# Logging Helpers
# ----------------------------
def log2str(pid, comment, logs):
    if type(logs) is str:
        logs = [logs]
    return ''.join([f"# JOB {pid} : --{comment}-- {log}\n" for log in logs])


def log_print(pid, comment, logs):
    text = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as f:
            f.write(text)
    if comment in [START, FINISH] and pid % 500 != 0:
        return
    print(text, end='')


# ----------------------------
# Image Cutting / Cropping
# ----------------------------
def cut_img(img, seq_info, frame_name, pid):
    if img.sum() <= 10000:
        msg = f"seq:{'-'.join(seq_info)}, frame:{frame_name}, no data, {img.sum()}."
        warn(msg)
        log_print(pid, WARNING, msg)
        return None

    y = img.sum(axis=1)
    y_top = (y != 0).argmax()
    y_btm = (y != 0).cumsum().argmax()
    img = img[y_top:y_btm + 1]

    ratio = img.shape[1] / img.shape[0]
    target_w = int(T_H * ratio)
    img = cv2.resize(img, (target_w, T_H), interpolation=cv2.INTER_CUBIC)

    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()

    x_center = np.searchsorted(sum_column, sum_point / 2)
    if x_center <= 0:
        msg = f"seq:{'-'.join(seq_info)}, frame:{frame_name}, no center."
        warn(msg)
        log_print(pid, WARNING, msg)
        return None

    half_w = T_W // 2
    left, right = x_center - half_w, x_center + half_w

    if left <= 0 or right >= img.shape[1]:
        pad = np.zeros((img.shape[0], half_w))
        img = np.concatenate([pad, img, pad], axis=1)
        left += half_w
        right += half_w

    return img[:, left:right].astype('uint8')


# ----------------------------
# Process a single video sequence
# ----------------------------
def cut_pickle(seq_info, pid):
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)

    log_print(pid, START, f"{'-'.join(seq_info)}")

    frame_list = sorted(os.listdir(seq_path))
    count = 0

    for frame_name in frame_list:
        frame_path = os.path.join(seq_path, frame_name)
        img = cv2.imread(frame_path)[:, :, 0]

        img = cut_img(img, seq_info, frame_name, pid)
        if img is not None:
            imageio.imwrite(os.path.join(out_dir, frame_name), img)
            count += 1

    if count < 5:
        msg = f"seq:{'-'.join(seq_info)}, less than 5 valid frames."
        warn(msg)
        log_print(pid, WARNING, msg)

    log_print(pid, FINISH, f"{count} valid frames saved → {out_dir}")


# ----------------------------
# Main Pretreatment Loop
# ----------------------------
print("Pretreatment Start.\n"
      f"Input path: {INPUT_PATH}\n"
      f"Output path: {OUTPUT_PATH}\n"
      f"Log file: {LOG_PATH}\n"
      f"Worker num: {WORKERS}\n")

pool = Pool(WORKERS)
results = []
pid = 0

id_list = sorted(os.listdir(INPUT_PATH))

for pid_folder in id_list:
    seq_types = sorted(os.listdir(os.path.join(INPUT_PATH, pid_folder)))

    for seq_type in seq_types:
        views = sorted(os.listdir(os.path.join(INPUT_PATH, pid_folder, seq_type)))

        for view in views:
            seq_info = [pid_folder, seq_type, view]
            os.makedirs(os.path.join(OUTPUT_PATH, *seq_info), exist_ok=True)

            results.append(pool.apply_async(cut_pickle, args=(seq_info, pid)))
            pid += 1
            sleep(0.02)

pool.close()

while True:
    unfinished = sum(1 for r in results if not r.ready())
    if unfinished == 0:
        break
    sleep(0.5)

pool.join()

