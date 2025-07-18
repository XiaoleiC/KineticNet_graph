from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import dgl

def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    Directly builds the adjacency matrix from a distance data frame.

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    # distances = dist_mx[~np.isinf(dist_mx)].flatten() / 1000 * 1.6
    # std = distances.std()
    # adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx = 1/(dist_mx / 1000 * 1.6 + 1e-8)  # Convert to km and add a small value to avoid division by zero.

    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='graph_sensor_ids.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='distances_la_2012.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].') # the unit is METER!!!
    parser.add_argument('--normalized_k', type=float, default=1/5,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='adj_mat.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    with open(args.sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
    normalized_k = args.normalized_k
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)
    np.save(args.output_pkl_filename.replace('.pkl', '.npy'), adj_mx, allow_pickle=True)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)

    src, dst = np.nonzero(adj_mx)
    weights = adj_mx[src, dst]
    g = dgl.graph((src, dst), num_nodes=adj_mx.shape[0])
    g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    bin_path = args.output_pkl_filename.replace('.pkl', '.bin')
    dgl.save_graphs(bin_path, [g])