
import graphtools
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors
import Helpers.utils as utils



class MAprocr():
    def __init__(self,
     r = 2,
     u = 1,
     knn = 5,
     decay=40,
     n_landmark=2000,
     t=10,
     gamma=1,
     n_pca=100,
     Uincluded = True,
     Dincluded = True,
     mds_solver="sgd",
     knn_dist="euclidean",
     knn_max=None,
     mds_dist="euclidean",
     mds="metric",
     n_jobs=1,
     random_state=None,
     verbose=0,
     potential_method=None,
     alpha_decay=None,
     njobs=None,
     k=None,
     a=None,
     distances="bila",
     diff_op_type = "nonsym",
     normalize_rw = True,
     ** kwargs):
    
        self.decay = decay
        self.knn = knn
        self.t = t
        self.n_landmark = n_landmark
        self.mds = mds
        self.n_pca = n_pca
        self.knn_dist = knn_dist
        self.knn_max = knn_max
        self.mds_dist = mds_dist
        self.mds_solver = mds_solver
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.graph = None
        self._diff_potential = None
        self.embedding = None
        self.X = None
        self.optimal_t = None
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.distances = distances
        self.diff_op_type = diff_op_type
        self.normalize_rw = normalize_rw
        self.r = r
        self.u = u
        self.Uincluded = Uincluded
        self.Dincluded = Dincluded
        
    def fit(self, domain1, domain2, sharedD1, sharedD2, labels1 = None, labels2 = None):
        
        self.domain1 = domain1
        self.domain2 = domain2
        self.sharedD1 = sharedD1
        self.sharedD2 = sharedD2
        
        self.domain1 = np.vstack((self.domain1, self.sharedD1))
        self.domain2 = np.vstack((self.domain2, self.sharedD2))

        self.Nshared = self.sharedD2.shape[0]
        
        self.N1 = self.domain1.shape[0]
        self.N2 = self.domain2.shape[0]


        indices_shared = np.arange(self.N1-self.Nshared, self.N1)
    
        self.graphD1 = graphtools.Graph(self.domain1,
                                        n_pca=self.n_pca,
                                        distance=self.knn_dist,
                                        knn=self.knn,
                                        knn_max=self.knn_max,
                                        decay=self.decay,
                                        thresh=1e-4,
                                        n_jobs=self.n_jobs,
                                        verbose=self.verbose,
                                        random_state=self.random_state,
                                        **(self.kwargs))
        self.graphD2 = graphtools.Graph(self.domain2,
                                        n_pca=self.n_pca,
                                        distance=self.knn_dist,
                                        knn=self.knn,
                                        knn_max=self.knn_max,
                                        decay=self.decay,
                                        thresh=1e-4,
                                        n_jobs=self.n_jobs,
                                        verbose=self.verbose,
                                        random_state=self.random_state,
                                        **(self.kwargs))
        
        
        # U = np.zeros((self.N1, self.N2))
        # U[indices_shared, indices_shared] = self.u
        
        # compute graph laplacian from similarities 
        W1 = self.graphD1.K.toarray()
        W2 = self.graphD2.K.toarray()
        D1 = np.diag(W1.sum(axis=0))
        D2 = np.diag(W2.sum(axis=0))
        self.L1 = D1 - W1
        self.L2 = D2 - W2
        
        # vl1, v1 = scipy.linalg.eigh(self.L1, D1)
        # vl2, v2 = scipy.linalg.eigh(self.L2, D2)
        # self.d1emb = v1[:,1:3]
        # self.d2emb = v2[:,1:3]
        # plt.scatter(self.d1emb[:,0], self.d1emb[:,1], c = domain1[:,1])
        # plt.figure()
        # plt.scatter(self.d2emb[:,0], self.d2emb[:,1], c = domain1[:,1])
        
        # pdb.set_trace()
        # D matrix 
        # Dz = np.block([[D1, np.zeros((self.N1, self.N2))], [np.zeros((self.N2, self.N1)), D2]])
        vl1, v1 = scipy.linalg.eigh(self.L1, D1)
        vl2, v2 = scipy.linalg.eigh(self.L2, D2)
        
        self.Al1_correspondences = v1[self.N1-self.Nshared:, 1:self.r]
        self.Al2_correspondences = v2[self.N2-self.Nshared:, 1:self.r]
        
        d, Z, tform = utils.procrustes(self.Al1_correspondences, self.Al2_correspondences, reflection = 'best')
        # 
        self.Al2_correspondences_ap = np.dot(tform['scale']*self.Al2_correspondences, 
                          tform['rotation']) + tform['translation']
        
        self.Al1 = v1[:self.N1-self.Nshared:, 1:self.r]
        self.Al2 = np.dot(tform['scale']*v2[:self.N2-self.Nshared, 1:self.r], 
                          tform['rotation']) + tform['translation']
        
        Al1_ = v1[:, 1:self.r]
        Al2_ = np.dot(tform['scale']*v2[:, 1:self.r], 
                          tform['rotation']) + tform['translation']
        
        # pdb.set_trace()
        self.Alignment = np.vstack((self.Al1, self.Al2))
        
        Alignment = np.vstack((Al1_, Al2_))
        
        self.graph_alignment = graphtools.Graph(Alignment,
                                n_pca=self.n_pca,
                                distance=self.knn_dist,
                                knn=self.knn,
                                knn_max=self.knn_max,
                                decay=self.decay,
                                thresh=1e-4,
                                n_jobs=self.n_jobs,
                                verbose=self.verbose,
                                random_state=self.random_state,
                                **(self.kwargs))
        
        self.W = self.graph_alignment.K.toarray()
        
        # assign neighbors to first  domain observations in the second domain
        nbrs = NearestNeighbors(n_neighbors=self.Al2.shape[0], algorithm='ball_tree').fit(self.Al2)
        _, indices = nbrs.kneighbors(self.Al1)
        a2 = indices[:,0]
        
        T = np.zeros((self.N1-self.Nshared,self.N2 - self.Nshared))
        T[range(self.N1-self.Nshared), a2] = 1
        
        return self.W #used to be t

