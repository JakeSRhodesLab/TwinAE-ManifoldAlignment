"""
ssma: Semi-Supervised Manifold Alignment

Semi-Supervised Manifold Alignment is a technique for aligning two data domains in a common low-dimensional space, with a focus on semi-supervised learning scenarios.

"""

import graphtools
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors

class ssma():
    def __init__(self,
                 r=2,
                 u=1000,
                 knn=5,
                 decay=40,
                 n_landmark=2000,
                 t=10,
                 gamma=1,
                 npca=100,
                 Lz_build=0,
                 Uincluded=False,
                 Dincluded=True,
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
                 diff_op_type="nonsym",
                 normalize_rw=True,
                 **kwargs):
        """
        Initialize the SSMA (Semi-Supervised Manifold Alignment) model.

        Parameters
        ----------
        r : int
            Dimensionality of the alignment space.
        u : int
            Weight for the shared points in the alignment.
        knn : int
            Number of nearest neighbors for constructing the graph.
        decay : float
            Decay factor for the affinity matrix.
        n_landmark : int
            Number of landmarks to select.
        t : int
            Number of iterations for the alignment algorithm.
        gamma : float
            Scaling factor for the alignment algorithm.
        npca : int
            Number of principal components to retain.
        Lz_build : int
            Option to build the graph laplacian Lz.
        Uincluded : bool
            Flag indicating whether to include U in Lz.
        Dincluded : bool
            Flag indicating whether to include D in Lz.
        mds_solver : str
            Solver to use for Multi-Dimensional Scaling (MDS).
        knn_dist : str
            Distance metric for k-nearest neighbors.
        knn_max : int
            Maximum number of neighbors to search for.
        mds_dist : str
            Distance metric for Multi-Dimensional Scaling (MDS).
        mds : str
            Type of Multi-Dimensional Scaling (metric or nonmetric).
        n_jobs : int
            Number of parallel jobs to run.
        random_state : int or None
            Seed for random number generation.
        verbose : int
            Verbosity level.
        potential_method : str
            Method for potential computation.
        alpha_decay : str
            Parameter for alpha decay.
        njobs : int
            Number of jobs for potential computation.
        k : int
            Parameter for potential computation.
        a : int
            Parameter for potential computation.
        distances : str
            Type of distances to use.
        diff_op_type : str
            Type of diffusion operator.
        normalize_rw : bool
            Whether to normalize random walk.
        **kwargs
            Additional keyword arguments for graph construction.

        """
        self.decay = decay
        self.knn = knn
        self.t = t
        self.n_landmark = n_landmark
        self.mds = mds
        self.npca = npca
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
        self.Lz_build = Lz_build

    def fit(self, domain1, domain2, sharedD1, sharedD2, labels1=None, labels2=None):
        """
        Fit the SSMA model to align two domains.

        Parameters
        ----------
        domain1 : numpy.ndarray
            Data points from the first domain.
        domain2 : numpy.ndarray
            Data points from the second domain.
        sharedD1 : numpy.ndarray
            Shared data points from the first domain.
        sharedD2 : numpy.ndarray
            Shared data points from the second domain.
        labels1 : numpy.ndarray or None, optional
            Labels for data points in domain1.
        labels2 : numpy.ndarray or None, optional
            Labels for data points in domain2.

        Returns
        -------
        numpy.ndarray
            Transformation matrix T representing the alignment between domain1 and domain2.
        """
        self.domain1u = domain1
        self.domain2u = domain2
        self.sharedD1 = sharedD1
        self.sharedD2 = sharedD2

        self.Nshared = self.sharedD2.shape[0]

        self.N1, self.f1 = self.domain1u.shape
        self.N2, self.f2 = self.domain2u.shape

        self.domain1 = np.vstack((self.domain1u, self.sharedD1))
        self.domain2 = np.vstack((self.domain2u, self.sharedD2))

        self.N1t = self.domain1.shape[0]
        self.N2t = self.domain2.shape[0]

        indices_shared = np.arange(self.N1t - self.Nshared, self.N1t)

        n_pca1 = np.minimum(self.npca, self.f1)
        if n_pca1 < 10:
            n_pca1 = None

        self.graphD1 = graphtools.Graph(self.domain1,
                                         n_pca=n_pca1,
                                         distance=self.knn_dist,
                                         knn=self.knn,
                                         knn_max=self.knn_max,
                                         decay=self.decay,
                                         thresh=1e-4,
                                         n_jobs=self.n_jobs,
                                         verbose=self.verbose,
                                         random_state=self.random_state,
                                         **(self.kwargs))

        n_pca2 = np.minimum(self.npca, self.f2)
        if n_pca2 < 10:
            n_pca2 = None

        self.graphD2 = graphtools.Graph(self.domain2,
                                         n_pca=n_pca2,
                                         distance=self.knn_dist,
                                         knn=self.knn,
                                         knn_max=self.knn_max,
                                         decay=self.decay,
                                         thresh=1e-4,
                                         n_jobs=self.n_jobs,
                                         verbose=self.verbose,
                                         random_state=self.random_state,
                                         **(self.kwargs))

        W1 = self.graphD1.K.toarray()
        W2 = self.graphD2.K.toarray()
        D1 = np.diag(W1.sum(axis=0))
        D2 = np.diag(W2.sum(axis=0))
        self.L1 = D1 - W1
        self.L2 = D2 - W2
        if self.Lz_build == 1:
            L1ll = self.L1[-self.Nshared:, -self.Nshared:]
            L1lu = self.L1[:self.N1, -self.Nshared:]
            L1ul = self.L1[-self.Nshared:, :self.N1]
            L1uu = self.L1[:self.N1, :self.N1]

            L2ll = self.L2[-self.Nshared:, -self.Nshared:]
            L2lu = self.L2[:self.N2, -self.Nshared:]
            L2ul = self.L2[-self.Nshared:, :self.N2]
            L2uu = self.L2[:self.N2, :self.N2]

            self.Lz = np.block([[L1ll + L2ll, L1ul, L2ul], [L1lu, L1uu, np.zeros((self.N1, self.N2))],
                                [L2lu, np.zeros((self.N2, self.N1)), L2uu]])

            if self.Dincluded:
                Dz = np.block([[D1[-self.Nshared:, -self.Nshared:] + D2[-self.Nshared:, -self.Nshared:],
                                np.zeros((self.Nshared, self.N1 + self.N2))],
                               [np.zeros((self.N1, self.Nshared)), D1[:self.N1, :self.N1],
                                np.zeros((self.N1, self.N2))],
                               [np.zeros((self.N1, self.Nshared)), np.zeros((self.N1, self.N2)),
                                D2[:self.N2, :self.N2]]])
                vl, v = scipy.linalg.eigh(self.Lz, Dz)
            else:
                vl, v = scipy.linalg.eigh(self.Lz)

            self.Al1 = v[self.Nshared:(self.N1 + self.Nshared), 1:self.r]
            self.Al2 = v[(self.N1 + self.Nshared):, 1:self.r]
            self.Alignment = v[:, 1:self.r]
        else:
            U = np.zeros((self.N1t, self.N2t))
            U[indices_shared, indices_shared] = self.u
            Dz = np.block([[D1, np.zeros((self.N1t, self.N2t))], [np.zeros((self.N2t, self.N1t)), D2]])
            if self.Uincluded:
                self.Lz = np.block([[self.L1 + U, -U], [-U.transpose(), self.L2 + U]])
            else:
                self.Lz = np.block([[self.L1, -U], [-U.transpose(), self.L2]])

            if self.Dincluded:
                vl, v = scipy.linalg.eigh(self.Lz, Dz)
            else:
                vl, v = scipy.linalg.eigh(self.Lz)

            self.Al1 = v[:self.N1t - self.Nshared, 1:self.r]
            self.Al2 = v[self.N1t:self.N1t + self.N2t - self.Nshared, 1:self.r]
            self.Alignment = v[:, 1:self.r]

        self.graph_alignment = graphtools.Graph(self.Alignment,
                                                 n_pca=n_pca1,
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

        nbrs = NearestNeighbors(n_neighbors=self.Al2.shape[0], algorithm='auto').fit(self.Al2)
        _, indices = nbrs.kneighbors(self.Al1)
        a2 = indices[:, 0]

        T = np.zeros((self.N1, self.N2))
        T[range(self.N1), a2] = 1
        return self.W #USED TO BE T