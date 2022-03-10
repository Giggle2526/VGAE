from json import load
from mimetypes import types_map
from sys import float_info

from matplotlib.pyplot import flag
from sklearn.covariance import log_likelihood
import args
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.metrics import adjusted_rand_score, log_loss, roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np

import model
from input_data import load_data
from preprocessing import *

adj, features = load_data(args.dataset)

adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix(
    (adj_orig.diagonal()[np.newaxis, :], [0]),
    shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, \
    val_edge_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

adj_norm = preproccess_graph(adj)
num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# pos_weight = Negtive / Positive
pos_weight = float(num_nodes ** 2 - adj.sum()) / adj.sum()
# norm = All / 2*Negtive
norm = num_nodes ** 2 / float(2 * (num_nodes ** 2 - adj.sum()))

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.sparse.FloatTensor(
    torch.LongTensor(adj_norm[0].T),
    torch.FloatTensor(adj_norm[1]),
    torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(
    torch.LongTensor(adj_label[0].T),
    torch.FloatTensor(adj_label[1]),
    torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(
    torch.LongTensor(features[0].T),
    torch.FloatTensor(features[1]),
    torch.Size(features[2]))

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight

model = getattr(model, args.model)(adj_norm)
optimizer = Adam(model.parameters(), lr=args.learning_rate)

def get_scores(edges_pos, edges_neg, adj_rec):
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds_pos = []
    pos = []
    for e in edges_pos:
        preds_pos.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].item()))
        neg.append(adj_orig[e[0], e[1]])
    
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

for epoch in range(args.num_epoch):
    t = time.time()
    
    A_pred = model(features)
    optimizer.zero_grad()
    loss = log_lik = norm*F.binary_cross_entropy(
        A_pred.view(-1), 
        adj_label.to_dense().view(-1),
        weight=weight_tensor)
    if args.model == 'VGAE':
        kl_divergence = 0.5/A_pred.size(0)*(
            torch.exp(model.log_var)+torch.square(model.mu)-model.log_var).sum(1).mean()
        loss += kl_divergence
    
    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred, adj_label)
    val_roc, val_ap = get_scores(val_edges, val_edge_false, A_pred)

    print(
        'Epoch: {ep:4d}, train_loss={tl:.5f}, train_acc={ta:.5f}, \
            val_roc={vr:.5f}, val_ap={va:.5f}, time={tm:.5f}'.format(
                ep=epoch+1, tl=loss.item(), ta=train_acc, 
                vr=val_roc, va=val_ap, tm=time.time()-t))

test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print('End of Training!\n Test_ROC={tr:.5f}, Test_AP={ta:.5f}'.format(tr=test_roc, ta=test_ap))

# GAE Result:
# Epoch:  200, train_loss=0.40843, train_acc=0.59432, val_roc=0.91736, val_ap=0.93585
# Test_ROC=0.90581, Test_AP=0.91629

# VGAE Result:
# Epoch:  200, train_loss=0.42580, train_acc=0.54924, val_roc=0.92098, val_ap=0.93333
# Test_ROC=0.91344, Test_AP=0.92283