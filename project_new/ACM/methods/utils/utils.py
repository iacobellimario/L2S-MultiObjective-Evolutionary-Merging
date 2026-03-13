import torch
import copy
import argparse
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import re

def calculate_variance(data):
    n = len(data)
    mean = sum(data) / n
    squared_diffs = [(x - mean) ** 2 for x in data]
    variance = sum(squared_diffs) / (n - 1)
    return variance

def copy_param_to_model(params: dict, model: nn.Module):
    for name, param in model.named_parameters():
        if name in params:
            param.data = params[name]
    return model

def copy_params_to_model(params1: dict, params2: dict, model: nn.Module):
    count = 0
    for name, param in model.named_parameters():
        number = re.findall(r'\d+', name)

        if name in params1:
            param.data = params1[name]
            count += 1
        if name in params2:
            param.data = params2[name]
            count += 1
    print('count:',count)
    return model

def compute_probabilities(digitized_signal, bins):
    prob = np.zeros(bins)
    for bin_index in digitized_signal:
        prob[bin_index - 1] += 1
    prob /= len(digitized_signal)
    return prob

def compute_joint_probabilities(X_dig, Y_dig, bins):
    joint_prob = np.zeros((bins, bins))
    for x_bin, y_bin in zip(X_dig, Y_dig):
        joint_prob[x_bin - 1, y_bin - 1] += 1
    joint_prob /= len(X_dig)
    return joint_prob

def mutual_information(P_X, P_Y, P_XY):
    mi = 0
    for x in range(len(P_X)):
        for y in range(len(P_Y)):
            if P_XY[x, y] > 0:
                mi += P_XY[x, y] * np.log(P_XY[x, y] / (P_X[x] * P_Y[y]))
    return mi

def discretize_signal(signal, bins):
    min_val, max_val = np.min(signal), np.max(signal)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    digitized = np.digitize(signal, bin_edges[:-1])
    return digitized

def entropy(array):
    _, counts = np.unique(array, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log(probabilities + 1e-10))

def list2numpy(tensor_list):
    import torch.nn.functional as F
    max_length = max(tensor.size(0) for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        padding = (0, max_length - tensor.size(0))
        padded_tensor = F.pad(tensor, padding, value=0)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.stack(padded_tensors)

    numpy_array = stacked_tensor.numpy()
    return numpy_array
