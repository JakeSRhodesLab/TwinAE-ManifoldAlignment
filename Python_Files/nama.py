# Nearest Anchor Manifold Alignment

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
from sklearn.neighbors  import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from icecream import ic
import ot
from scipy import sparse


from sklearn.manifold import MDS
import seaborn as sns


import ot

# TODO: Add OT functionality, using kernel instead of distance?
# TODO: Add plotting functions, e.g., matrix visual
# TODO: Add a function to test a datset; gets random subset of anchors,
        # Makes a duplicate dataset with noise
class NAMA():
    """Nearest-Anchor Manifold Alignment
    """


    def __init__(self, ot_reg = 0):

        """
        Parameters
        ----------
        known_anchors: array-like or list
            A list of indices of known-correspondences. Or, a list of tuples indicating the pairwise known correpondences of x, y.


        Returns
        -------
        
        """

        self.ot_reg = ot_reg


    def compute_distances(self, x, y):
        x_dists = squareform(pdist(x))
        self.x_dists = x_dists / np.max(x_dists, axis = None)

        y_dists = squareform(pdist(y))
        self.y_dists = y_dists / np.max(y_dists, axis = None)

        self.x_nns = np.argsort(x_dists)
        self.y_nns = np.argsort(y_dists)


    def format_anchors(self, known_anchors):
        if np.ndim(known_anchors) == 1:
            known_anchors = np.vstack(known_anchors)
        
        self.known_anchors = known_anchors
        self.x_anchors = known_anchors[:, 0]
        self.y_anchors = known_anchors[:, 1]

        # Do we want to return something or just asign to self?
        # return known_anchors

    def find_neighbors(self):
        pass

    def get_block(self, cross_domain_dists):
        self.block = np.block([[self.x_dists, cross_domain_dists],
                  [cross_domain_dists.T, self.y_dists]])
        

    def get_optimal_transport(self, a, b, D):
        if self.ot_reg <= 0:
            self.T = ot.emd(a, b, D)

        else:
            self.T = ot.sinkhorn(a, b, D, reg = self.ot_reg)

        return self.T



    def fit(self, known_anchors, x, y, link_agg = 'mean'):
        # TODO: Try writing in the form of a graph
        # TODO: Make sure this works with datasets of different dims

        self.nx, self.dx = np.shape(x)
        self.ny, self.dy = np.shape(y)
        self.format_anchors(known_anchors)

        self.compute_distances(x, y)
        self.cross_domain_dists = np.ones((self.nx, self.ny))


        # Need to account for nearest neighbors, avoid jumps to anchors

        for i in range(self.nx):
            xi_nn = self.x_nns[i, :]
            for j in range(self.ny):
                yj_nn = self.y_nns[j, :]
                # if i >= j: # TODO: Check this with other values...
                if [i, j] in self.known_anchors.tolist():
                    self.cross_domain_dists[i, j] = 0
                    self.cross_domain_dists[j, i] = 0
                elif i in self.x_anchors:
                    idx = np.where(self.x_anchors == i)[0][0]
                    yi = self.known_anchors[idx, 1]
                    self.cross_domain_dists[i, j] = self.y_dists[yi, j]
                    self.cross_domain_dists[j, i] = self.y_dists[yi, j]
                elif j in self.y_anchors:
                    idx = np.where(self.y_anchors == j)[0][0]
                    xj = self.known_anchors[idx, 0]
                    self.cross_domain_dists[i, j] = self.x_dists[i, xj]
                    self.cross_domain_dists[j, i] = self.x_dists[i, xj]
                else:
                    # Index to then nearest link in X

                    xk_list = [np.where(arg == xi_nn)[0][0] for arg in self.x_anchors]
                    xk_list.sort()
                    xk = xk_list[0]


                    # What is the nearest link in domain X
                    xi_link = xi_nn[xk]
                    yi_link = self.known_anchors[np.where(xi_link == self.known_anchors[:, 0])[0][0], 1]


                    # The distance to the nearest link in X
                    dist_xk = self.x_dists[i, xi_link]
                    dist_yk = self.y_dists[j, yi_link]


                    # The nearest link in Y
                    yt_list = [np.where(arg == yj_nn)[0][0] for arg in self.y_anchors]
                    yt_list.sort()
                    yt = yt_list[0]


                    # What is the nearest link in domain Y
                    yj_link = yj_nn[yt]
                    xj_link = self.known_anchors[np.where(yj_link == self.known_anchors[:, 1])[0][0], 0]


                    # # The distance to the nearest link in Y
                    dist_yt = self.y_dists[j, yj_link]
                    dist_xt = self.x_dists[i, xj_link]


                    if link_agg == 'mean':
                        self.cross_domain_dists[i, j] = np.mean([dist_xk, dist_yk, dist_xt, dist_yt])
                    elif link_agg == 'mean_sum':
                        self.cross_domain_dists[i, j] = np.mean([dist_xk, dist_yk, dist_xt, dist_yt])
                    elif link_agg == 'min':
                        self.cross_domain_dists[i, j] = np.min([dist_xk, dist_yk, dist_xt, dist_yt])
                    elif link_agg == 'max':
                        self.cross_domain_dists[i, j] = np.max([dist_xk, dist_yk, dist_xt, dist_yt])
                    else:
                        raise NameError('link_agg must be mean, mean_sum, min, or max.')

                    self.cross_domain_dists[j, i] = self.cross_domain_dists[i, j]

        self.get_optimal_transport(a = np.ones((self.nx, ), dtype = 'int8'), b = np.ones((self.ny, ), dtype = 'int8'), D = self.cross_domain_dists)

        self.get_block(self.cross_domain_dists)

        return self.block

        



            

