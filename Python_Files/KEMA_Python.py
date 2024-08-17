"""
KEMA: Kernal Manifold Alignment -- Supervised

Original MATLAB code: 
https://github.com/dtuia/KEMA/blob/master/general_routine/KMA.m

"""


import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import eigh
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import block_diag

def KMA(labeled, unlabeled, options, debug = False):

    #Convert to numpy array
    unlabeled = np.array(unlabeled)
    labeled = np.array(labeled)

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

    n = 0
    d = 0

    X_list = []  # To store combined matrices

    if debug:
        print(f"Number of Domains: {options['numDomains']}")
        
        print(f"First Labeled Information: {labeled[0, 0]}")
        print(f"First Unlabeled Information: {unlabeled[0, 0]}")
        print("<><><><><>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")


    for data in unlabeled:
        #The code for some reason has it backwards
        X_list.append(data.T)

    #Make one big list with labels
    Y = labeled[0]
    for labels in labeled[1:]:
        Y = np.hstack([Y, labels])


    # Building Laplacians
    W = []

    for X in X_list:
        G = kneighbors_graph(X.T, options['nn'], mode='connectivity')
        W.append(G)

    W = block_diag(W)
    W = W.toarray()

    if debug:
        from matplotlib.pyplot import imshow, show
        print("-----------------   The W matrix   -----------------------")
        imshow(W), show()
        input("Continue? Press Enter.")

    # Class Graph Laplacian
    Ws = (np.tile(Y, (len(Y), 1)) == np.tile(Y, (len(Y), 1)).T).astype(float)
    Ws[Y == 0, :] = 0
    Ws[:, Y == 0] = 0

    if debug:
        print("-----------------   The Ws matrix   -----------------------")
        imshow(Ws), show()
        input("Continue? Press Enter.")

    Wd = (np.tile(Y, (len(Y), 1)) != np.tile(Y, (len(Y), 1)).T).astype(float)
    Wd[Y == 0, :] = 0
    Wd[:, Y == 0] = 0

    if debug:
        print("-----------------   The Wd matrix   -----------------------")
        imshow(Ws), show()
        input("Continue? Press Enter.")

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

    if debug:
        print("KAK MATRIX")
        imshow(KAK),show()

        input("Continue? Press Enter.")

        print("KBK MATRIX")
        imshow(KBK), show()
        input("Continue? Press Enter.")

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
