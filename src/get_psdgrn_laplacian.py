import numpy as np
import scipy.sparse as sp
import torch


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def cheb_poly_sparse(A, K):
    K += 1
    N = A.shape[0]
    multi_order_laplacian = [
        sp.eye(N, dtype=np.float32, format="coo")
    ]

    if K == 1:
        return multi_order_laplacian

    multi_order_laplacian.append(A)
    if K == 2:
        return multi_order_laplacian

    for k in range(2, K):
        multi_order_laplacian.append(
            2.0 * A.dot(multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]
        )
    return multi_order_laplacian


def hermitian_decomp_sparse(
    pos_edges,
    neg_edges,
    size,
    q=0.25,
    norm=True,
    laplacian=True,
    max_eigen=2.0
):
    pos_edges = np.asarray(pos_edges, dtype=np.int64).reshape(-1, 2)
    neg_edges = np.asarray(neg_edges, dtype=np.int64).reshape(-1, 2)

    pos_row = pos_edges[:, 0] if len(pos_edges) > 0 else np.array([], dtype=np.int64)
    pos_col = pos_edges[:, 1] if len(pos_edges) > 0 else np.array([], dtype=np.int64)

    neg_row = neg_edges[:, 0] if len(neg_edges) > 0 else np.array([], dtype=np.int64)
    neg_col = neg_edges[:, 1] if len(neg_edges) > 0 else np.array([], dtype=np.int64)

    A = sp.coo_matrix(
        (
            np.ones(len(pos_row) + len(neg_row), dtype=np.float32),
            (
                np.concatenate([pos_row, neg_row]),
                np.concatenate([pos_col, neg_col])
            )
        ),
        shape=(size, size),
        dtype=np.float32
    )

    diag = sp.eye(size, dtype=np.float32, format="coo")
    A_sym = 0.5 * (A + A.T)

    if norm:
        d = np.array(A_sym.sum(axis=0)).reshape(-1)
        d[d == 0] = 1.0
        d = np.power(d, -0.5)
        D = sp.diags(d)
        A_sym = D.dot(A_sym).dot(D)

    if laplacian:
        phase_pos = sp.coo_matrix(
            (np.ones(len(pos_row)), (pos_row, pos_col)),
            shape=(size, size),
            dtype=np.float32
        )
        theta_pos = q * 1j * phase_pos
        theta_pos.data = np.exp(theta_pos.data)
        theta_pos_t = -q * 1j * phase_pos.T
        theta_pos_t.data = np.exp(theta_pos_t.data)

        phase_neg = sp.coo_matrix(
            (np.ones(len(neg_row)), (neg_row, neg_col)),
            shape=(size, size),
            dtype=np.float32
        )
        theta_neg = (np.pi + q) * 1j * phase_neg
        theta_neg.data = np.exp(theta_neg.data)
        theta_neg_t = (np.pi - q) * 1j * phase_neg.T
        theta_neg_t.data = np.exp(theta_neg_t.data)

        data = np.concatenate([
            theta_pos.data, theta_pos_t.data,
            theta_neg.data, theta_neg_t.data
        ])
        theta_row = np.concatenate([
            theta_pos.row, theta_pos_t.row,
            theta_neg.row, theta_neg_t.row
        ])
        theta_col = np.concatenate([
            theta_pos.col, theta_pos_t.col,
            theta_neg.col, theta_neg_t.col
        ])

        Theta = sp.coo_matrix(
            (data, (theta_row, theta_col)),
            shape=(size, size),
            dtype=np.complex64
        )

        # Align with SD-GCN Eq. (3): normalize the summed complex phase
        # for each node pair after reciprocal-edge contributions are merged.
        Theta = Theta.tocsr()
        Theta.sum_duplicates()
        phase_eps = 1e-5
        Theta.data = Theta.data / (np.abs(Theta.data) + phase_eps)
        Theta.eliminate_zeros()
        Theta = Theta.tocoo()

        if norm:
            D = diag
        else:
            d = np.array(A_sym.sum(axis=0)).reshape(-1)
            D = sp.diags(d)

        L = D - Theta.multiply(A_sym)
        L = (2.0 / max_eigen) * L - diag
        return L

    return A_sym


def build_psdgrn_laplacian(pos_edges, neg_edges, num_nodes, K, q, device, norm=True):
    L = hermitian_decomp_sparse(
        pos_edges=pos_edges,
        neg_edges=neg_edges,
        size=num_nodes,
        q=q,
        norm=norm,
        laplacian=True,
        max_eigen=2.0
    )

    multi_L = cheb_poly_sparse(L, K)

    L_real = []
    L_imag = []
    for Lk in multi_L:
        L_real.append(sparse_mx_to_torch_sparse_tensor(Lk.real, device))
        L_imag.append(sparse_mx_to_torch_sparse_tensor(Lk.imag, device))

    return L_real, L_imag