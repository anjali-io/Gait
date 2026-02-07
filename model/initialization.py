# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15

import os
from copy import deepcopy
import numpy as np

from .utils import load_data
from .model import Model


def initialize_data(config):
    """
    Initialize training and testing datasets.
    Uses PRETREATED data only (CASIA-B-64).
    """
    print("Initializing data source...")
    train_source, test_source = load_data(**config['data'], cache=True)
    print("Data initialization complete.")
    return train_source, test_source


def initialize_model(config, train_source, test_source):
    """
    Initialize model with all required parameters (including CVL).
    """
    print("Initializing model...")

    data_config = config['data']
    model_config = config['model']

    # Copy model parameters safely
    model_param = deepcopy(model_config)

    # Inject required runtime parameters
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_config['pid_num']

    # Build save name
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str, [
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    # Initialize model
    model = Model(**model_param)

    print("Model initialization complete.")
    return model, model_param['save_name']


def initialization(config):
    """
    Entry point for training / testing.
    """
    print("Initializing environment...")

    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

    train_source, test_source = initialize_data(config)
    return initialize_model(config, train_source, test_source)
