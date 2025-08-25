import random
import h5py
import numpy as np
import sympy
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math
import torchvision
import pandas as pd
import glob
import os
import sys
from scipy.spatial.distance import squareform
import scipy as sp
import networkx
import torch.distributed as dist
import time
import timeit
from datetime import date
import struct
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.MNISTModel import MNISTModel
from model.CIFAR10Model import CIFAR10Model
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import RandomState, SeedSequence
from numpy.random import MT19937
from model.model import *
from model.CIFAR10Model import *
from sklearn.cluster import KMeans
import itertools
import json
from sympy import Symbol
from sympy.solvers.inequalities import reduce_rational_inequalities
from trans_matrix import *


def moving_average(input_data, window_size):
    moving_average = [[] for i in range(len(input_data))]
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if j < window_size - 1:
                if type(input_data[i][j + 1]) == str:
                    input_data[i][j + 1] = float(input_data[i][j + 1])
                if input_data[i][j + 1] == 'nan' or 'inf':
                    input_data[i][j + 1] = float(input_data[i][j])
                moving_average[i].append(sum(input_data[i][:j + 1]) / len(input_data[i][:j + 1]))
            else:
                input_data[i][j - window_size + 1:j + 1][-1] = float(input_data[i][j - window_size + 1:j + 1][-1])
                # print(input_data[i][j - window_size + 1:j + 1])
                moving_average[i].append(sum(input_data[i][j - window_size + 1:j + 1]) / len(input_data[i][j - window_size + 1:j + 1]))
    moving_average_means = []
    for i in range(len(moving_average[0])):
        sum_data = []
        for j in range(len(moving_average)):
            sum_data.append(moving_average[j][i])
        moving_average_means.append(sum(sum_data) / len(sum_data))
    return np.array(moving_average), moving_average_means

def matrix(nodes, num_neighbor):
    upper = int(nodes / 2) - 2
    bottom = 1
    matrix = np.ones((nodes,), dtype=int)
    while True:
        org_matrix = np.diag(matrix)
        org_target = np.arange(nodes, dtype=int)
        for i in range(nodes):
            if np.count_nonzero(org_matrix[i]) < num_neighbor + 1:
                if np.count_nonzero(org_matrix[i]) < num_neighbor + 1 and np.count_nonzero(
                        org_matrix.transpose()[i]) < num_neighbor + 1:
                    target = np.setdiff1d(org_target, i)
                    target_set = []
                    for k in range(len(target)):
                        if np.count_nonzero(org_matrix[target[k]]) < num_neighbor + 1:
                            target_set.append(target[k])
                    if num_neighbor + 1 - int(np.count_nonzero(org_matrix[i])) <= len(target_set):
                        target = np.random.choice(target_set, num_neighbor + 1 - int(np.count_nonzero(org_matrix[i])),
                                                  replace=False)
                    for j in range(len(target)):
                        org_matrix[i][target[j]] = 1
                        org_matrix.transpose()[i][target[j]] = 1
            else:
                pass
        if np.count_nonzero(
                np.array([np.count_nonzero(org_matrix[i]) for i in range(nodes)]) - (num_neighbor + 1)) == 0:
            break
    return org_matrix

def Ring_network(nodes):
    matrix = np.ones((nodes,), dtype=int)
    conn_matrix = np.diag(matrix)
    neighbors = []
    for i in range(nodes):
        connected = [(i - 1) % nodes, i, (i + 1) % nodes]
        for j in connected:
            conn_matrix[i][j] = 1
            conn_matrix.transpose()[i][j] = 1
        neighbors.append(connected)
    factor = 1 / len(neighbors[0])
    conn_matrix = conn_matrix * factor
    return conn_matrix,

def Check_Matrix(client, matrix):
    count = 0
    for i in range(client):
        if np.count_nonzero(matrix[i] - matrix.transpose()[i]) == 0:
            pass
        else:
            count += 1
    if count != 0:
        raise Exception('The Transfer Matrix Should be Symmetric')
    else:
        print('Transfer Matrix is Symmetric Matrix')


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'arial'

num_nodes = 6
num_neighbors = 2
num_classes = 10
seed = 24

# if num_neighbors == 2:
#     network = 'Ring'
# else:
#     network = 'Random'
#
# conn_matrix = Transform(num_nodes=num_nodes, num_neighbors=num_neighbors, seed=seed, network=network)
# print('W: ', conn_matrix.matrix)
# Check_Matrix(client=num_nodes, matrix=conn_matrix.matrix)
#
# neighbors = list(conn_matrix.neighbors)
# print('Neighbors: ', neighbors)
#
# neighbors_weights = [[0 for i in range(num_neighbors+1)] for i in range(num_nodes)]
# print('Initial neighbor weights: ', neighbors_weights)
# for n in range(num_nodes):
#     node_weights = n
#     for m in range(num_nodes):
#         if n in neighbors[m]:
#             neighbors_weights[m][neighbors[m].index(n)] = node_weights
# print('Updated neighbor weights: ', neighbors_weights)

alpha = 0.01
Alpha = [alpha for i in range(num_classes)]
samples = np.random.dirichlet(Alpha, size=num_nodes)
print(len(samples), samples)
summation = np.sum(samples, axis=1)
print(summation)
for i in range(len(samples)):
    data_samples = np.array(6000 * samples[i], dtype=np.int16)
    print(data_samples, sum(data_samples))
