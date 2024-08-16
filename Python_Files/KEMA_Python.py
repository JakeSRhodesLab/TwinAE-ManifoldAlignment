"""
KEMA: Kernal Manifold Alignment -- Supervised
Original MATLAB code: https://github.com/dtuia/KEMA

"""


import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import eigh
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import block_diag

def KMA(labeled, unlabeled, options):
    # Set default options if not provided
    options.setdefault('numDomains', len(labeled))
    options.setdefault('kernelt', 'lin')
    
    if options['kernelt'] == 'rbf' and 'sigma' not in options:
        print('Estimating sigma from data of domain 1')
        options['sigma'] = [np.mean(pdist(domain['X'].T)) for domain in labeled]
    elif options['kernelt'] == 'pga' and 'sigma' not in options:
        print('Estimating sigma from data of domain 1')
        options['sigma'] = [15 * np.mean(pdist(domain['X'].T)) for domain in labeled]
        options.setdefault('b', 2)

    options.setdefault('nn', 9)
    options.setdefault('mu', 0.5)

    # Create data matrices
    Y = np.array([])
    n = 0
    d = 0
    X_list = []
    Y_list = []

    for domain in labeled:
        X_combined = np.hstack([domain['X'], unlabeled[labeled.index(domain)]['X']])
        X_list.append(X_combined)
        Y_combined = np.hstack([domain['Y'], np.zeros(unlabeled[labeled.index(domain)]['X'].shape[1])])
        Y_list.append(Y_combined)
        Y = np.hstack([Y, Y_combined])
        n += X_combined.shape[1]
        d += X_combined.shape[0]

    # Building Laplacians
    W = []

    for X in X_list:
        G = kneighbors_graph(X.T, options['nn'], mode='connectivity')
        W.append(G)

    W = block_diag(W)
    W = W.toarray()

    # Class Graph Laplacian
    Ws = (np.tile(Y, (len(Y), 1)) == np.tile(Y, (len(Y), 1)).T).astype(float)
    Ws[Y == 0, :] = 0
    Ws[:, Y == 0] = 0
    Wd = (np.tile(Y, (len(Y), 1)) != np.tile(Y, (len(Y), 1)).T).astype(float)
    Wd[Y == 0, :] = 0
    Wd[:, Y == 0] = 0

    # Normalization
    Sws = Ws.sum()
    Sw = W.sum()
    Ws = Ws / Sws * Sw

    Swd = Wd.sum()
    Wd = Wd / Swd * Sw

    Ds = np.sum(Ws, axis=1)
    Ls = np.diag(Ds) - Ws

    D = np.sum(W, axis=1)
    L = np.diag(D) - W

    Dd = np.sum(Wd, axis=1)
    Ld = np.diag(Dd) - Wd

    A = ((1 - options['mu']) * L + options['mu'] * Ls) + options.get('lambda', 0) * np.eye(len(Ls))
    B = Ld

    # Kernel computation
    if options['kernelt'] in ['wan', 'lin']:
        Z = np.block([X for X in X_list])
        K = Z.T @ Z
    else:
        K = np.zeros((n, n))
        for i, X in enumerate(X_list):
            # Kernel matrix computations (assuming an RBF kernel for now)
            if options['kernelt'] == 'rbf':
                sigma = options['sigma'][i]
                K_i = np.exp(-pdist(X.T, 'sqeuclidean') / (2 * sigma ** 2))
                K += block_diag([K_i])
    
    KAK = K @ A @ K
    KBK = K @ B @ K

    # Solve generalized eigenvalue problem
    ALPHA, LAMBDA = eigh(KAK, KBK)

    # Sort eigenvalues and corresponding eigenvectors
    idx = np.argsort(LAMBDA)
    LAMBDA = LAMBDA[idx]
    ALPHA = ALPHA[:, idx]

    # Limit number of projections if necessary
    if ALPHA.shape[1] < options['projections']:
        options['projections'] = ALPHA.shape[1]
        print(f'Reduced the number of projections to {options["projections"]}.')
    
    return ALPHA, LAMBDA, options
