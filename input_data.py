from isort import file
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open('data/ind.{}.{}'.format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)

    test_idx = parse_index_file('data/ind.{}.test.index'.format(dataset))
    test_idx_reorder = np.sort(test_idx)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx), max(test_idx)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full)), x.shape[1])
        tx_extended[test_idx_reorder-min(test_idx_reorder), :] = tx
        tx = tx_extended
    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_reorder, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    return adj, features