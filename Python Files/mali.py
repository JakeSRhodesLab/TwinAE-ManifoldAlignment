
import graphtools
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import eigs, norm
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import ot
from vne import find_optimal_t
from rfgap import RFGAP
import scipy
from icecream import ic
from utils import col_normalize, row_normalize, label_impute
from loguru import logger

# TODO: Set up for more than 2 blocks
# TODO: kwargs not currently used


class MALI():
    # TODO: add landmarking at graph level
    # Set t: auto to VNE selection

    def __init__(self, graph_distance = 'euclidean', interclass_distance = 'cosine', knn = 5,
                 n_pca = 100, mu = 0.5, t = 'auto', transition_only = False, ot = True, ot_reg = 0.001, a = 'class_priors', b = 'uniform', class_prior_inverse = True, graph_knn = 10, graph_decay = 10, normalize_M = True, random_state = None):
        self.graph_distance = graph_distance
        self.interclass_distance = interclass_distance
        self.knn = knn
        self.n_pca = n_pca
        self.mu = mu
        self.t = t
        self.transition_only = transition_only
        self.ot = ot
        self.ot_reg = ot_reg
        self.a = a
        self.b = b
        self.graph_knn = graph_knn
        self.graph_decay = graph_decay
        self.class_prior_inverse = class_prior_inverse
        self.normalize_M = normalize_M
        self.random_state = random_state        


    def build_graph(self, x, distance = 'euclidean'):

        if self.n_pca is None:
            return graphtools.Graph(data = x, distance = distance, knn = self.graph_knn,
                        decay = self.graph_decay)
        
        else:
            n_pca = min(self.n_pca, np.shape(x)[0])
            return graphtools.Graph(data = x, n_pca = n_pca, distance = distance, knn = self.graph_knn,
                        decay = self.graph_decay)


    

    def accumulated_transition(self, P):

        if self.t == 'auto':
            self.optimal_t = find_optimal_t(P.todense()) # TODO: May not want to store optimal_t here; may be different for different domains
            print('optimal t: ', self.optimal_t)
            return P ** self.optimal_t

        elif self.t == 'auto-I':
            self.optimal_t = find_optimal_t(P.todense()) # TODO: May not want to store optimal_t here; may 
            print('optimal t: ', self.optimal_t)
            I = np.eye(P.shape[0])
            return P ** self.optimal_t - I # I still works without missing; higher self-affinity scores

        
        elif self.t == 'DPT': # diffusion pseudotime
            phi      = np.real(eigs(P, k = 1)[1])
            ones     = np.ones_like(phi)
            I        = np.eye(np.shape(ones)[0])
            phi_ones = np.outer(phi, np.transpose(phi))
            return np.linalg.inv((I - (P - phi_ones))) - I
        
        elif self.t == 'DPT-I': # diffusion pseudotime
            phi      = np.real(eigs(P, k = 1)[1])
            ones     = np.ones_like(phi)
            I        = np.eye(np.shape(ones)[0])
            phi_ones = np.outer(phi, np.transpose(phi))
            return np.linalg.inv((I - (P - phi_ones)))
        
        else:
            return P ** self.t
        
    
    def class_priors(self, y, inverse = True):
        # TODO: Use some sort of soft-max approach?
        n = len(y)
        hash_a = {label: np.sum(y == label) for label in y}
        a_counts = np.ones_like(y, dtype = np.float64)

        for (key, value) in hash_a.items():
            a_counts[y == key] = value
      

        a_vals = a_counts * (n / sum(a_counts))

        if inverse:
            a_vals = (1 / a_counts) * (n / len(hash_a)) 

        return a_vals


    def class_transition(self, y, y_intersect, M):

        n = len(y)
        unique_y = sorted(list(set(y_intersect[y_intersect != -1])))

        priors   = np.zeros_like(unique_y, dtype= np.float32)
        Ml = np.zeros((n, len(unique_y)))
        for j, label in enumerate(unique_y):
            priors[j] = (np.count_nonzero(y == label) / n)
            Ml[:, j] = np.sum(M[:, y == label], axis = 1).squeeze() / priors[j]

        return Ml


    def class_distance(self, Ml1, Ml2):
        return pairwise_distances(Ml1, Ml2, self.interclass_distance)
            

    def get_nearest(self, n1, n2, D):
        nearest_idx = np.argpartition(D, kth = self.knn, axis = 1)[:, :self.knn]
        T = np.zeros((n1, n2))

        for i in range(n1):
            # T[i, nearest_idx[i, :]] = 1 # TODO: Should this be 1 / self.knn?
            T[i, nearest_idx[i, :]] = 1 / self.knn # Doesn't guarantee at least one non-zero in col/row

        self.T = sparse.csr_matrix(T)
        return self.T
    

    def get_optimal_transport(self, a, b, D):
        if self.ot_reg <= 0:
            self.T = ot.emd(a, b, D)

        else:
            self.T = ot.sinkhorn(a, b, D, reg = self.ot_reg)

        self.T = sparse.csr_matrix(self.T)
        return self.T


    def cross_domain_similarity(self, T, W1, W2):
        if self.transition_only:
            return T
        
        else:
            W_cross = W1 @ T + T @ W2
            return W_cross 
        

    def extend_similarity(self, T, Wx, normalize_columns = False):
        if self.transition_only:
            return T
        
        if normalize_columns:
            return Wx @ col_normalize(T)
        else:
            return np.transpose(row_normalize(T) @ Wx)
    

            
    @logger.catch
    def fit(self, X, Y):
        """
        X, Y are tuples; each entry a dataset or set of labels
        """
        ic()

        ns = [np.shape(x)[0] for x in X]
        graphs = {}
        Ws     = {}
        Ms     = {}
        Mls    = {}

        y_intersect = np.intersect1d(Y[0], Y[1])


        for i, (x, y) in enumerate(zip(X, Y)):

            if self.graph_distance == 'rfgap':
                rf = RFGAP(y = y, non_zero_diagonal = True, random_state = self.random_state,
                           oob_score = True, prox_method = 'rfgap') 

                y, rf = label_impute(x, y, rf)
                rf.fit(x, y)
                self.oob_score_ = rf.oob_score_

                graphs[i] = rf
                Ws[i] = row_normalize(rf.get_proximities())
                Ms[i] = self.accumulated_transition(Ws[i])


            else:

                graphs[i] = self.build_graph(x, self.graph_distance)
                Ws[i]     = graphs[i].kernel
                Ms[i]     = self.accumulated_transition(graphs[i].diff_op)

            if self.normalize_M:
                scaler = MinMaxScaler()
                try:
                    Ms[i] = scaler.fit_transform(np.asarray((Ms[i].todense())))
                except:
                    Ms[i] = scaler.fit_transform(np.asarray((Ms[i])))
        

            Mls[i] = self.class_transition(y, y_intersect, Ms[i])

        self.Ms = Ms
        self.Mls = Mls



        if self.interclass_distance == 'rfgap':

            model = RFGAP() # TODO: If needed, make more flexible
            y2 = Y[1]
            y2[y2 == -1] = np.argmax(Mls[1][y2 == -1, :], axis = 1)
            model.fit(np.vstack([Mls[0], Mls[1]]), np.hstack([Y[0], y2]))

            D_rfgap = np.asarray(model.get_proximities().todense())
            D = np.max(D_rfgap[len(Y[0]):, :len(Y[1])]) - D_rfgap[len(Y[0]):, :len(Y[1])] # To make into distance

            self.D = D 
            self.D_rfgap = D_rfgap
            
        else:
            D = self.class_distance(Mls[0], Mls[1])
            self.D = D

        if self.ot:
            if self.a == 'uniform':
                self.a_dist = np.ones_like(Y[0], dtype = 'int8')

            elif self.a == 'class_priors':
                self.a_dist = self.class_priors(Y[0], inverse = self.class_prior_inverse)
            
            if self.b == 'uniform':
                self.b_dist = np.ones_like(Y[1], dtype = 'int8')

            elif self.b == 'class_priors':
                self.b_dist = self.class_priors(Y[1], inverse = self.class_prior_inverse)

            T = self.get_optimal_transport(self.a_dist, self.b_dist, D)


        else:
            T = self.get_nearest(ns[0], ns[1], D)

        
        # TODO: Write argument to keep original MALI implementation?
        #---------------------------------------------------------------------------#
        #                         Updated use of T
        #---------------------------------------------------------------------------#
        self.Wxy = self.extend_similarity(T, Ws[0], normalize_columns = True)
        self.Wyx = self.extend_similarity(T, Ws[1])

        W_cross = (self.Wxy + np.transpose(self.Wyx)) / 2
        self.W_cross = W_cross
        #--------------------------------------------------------------------------#


        self.W = sparse.bmat([[self.mu * Ws[0], (1 - self.mu) * W_cross],
            [(1 - self.mu) * sparse.csr_matrix.transpose(W_cross), self.mu * Ws[1]]])

        self.Ws = Ws
        self.W_cross = W_cross
        self.graphs = graphs
        self.T = T 


    def fit_transform():
        pass




