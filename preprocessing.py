import numpy as np
import scipy.sparse as sp

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preproccess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_inv_sqrt).transpose().dot(degree_inv_sqrt)
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
    adj = adj - sp.dia_matrix(
        (adj.diagonal()[np.newaxis, :], [0]), 
        shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    edges = sparse_to_tuple(adj_triu)[0]
    edges_all = sparse_to_tuple(adj)[0]

    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    test_edge_idx = all_edge_idx[:num_test]
    test_edges = edges[test_edge_idx]
    val_edge_idx = all_edge_idx[num_test:(num_test+num_val)]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(
        edges, 
        np.hstack([test_edge_idx, val_edge_idx]), 
        axis=0)
    
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        e = [[idx_i, idx_j], [idx_j, idx_i]]
        if idx_i == idx_j: continue
        if ismember(e, edges_all): continue
        if test_edges_false:
            if ismember(e, np.array(test_edges_false)): continue
        test_edges_false.append(e[0])   # single-direction
    
    val_edge_false = []
    while len(val_edge_false) < len(val_edge_idx):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        e = [[idx_i, idx_j], [idx_j, idx_i]]
        if idx_i == idx_j: continue
        if ismember(e, train_edges): continue
        if ismember(e, val_edges): continue
        if val_edge_false:
            if ismember(e, np.array(val_edge_false)): continue
        val_edge_false.append(e[0])     # single-direction
    
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edge_false, edges_all)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    adj_train = sp.csr_matrix(
        (data, (train_edges[:, 0], train_edges[:, 1])),
        shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, val_edges, \
        val_edge_false, test_edges, test_edges_false