"""
KEMA: Kernal Manifold Alignment -- Supervised

Original MATLAB code: 
https://github.com/dtuia/KEMA/blob/master/general_routine/KMA.m

"""


import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import eigh
from sklearn.neighbors import kneighbors_graph, NearestNeighbors, KNeighborsClassifier
from scipy.sparse import block_diag
from sklearn.manifold import MDS
from pandas import Categorical
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import block_diag as blkdiag

class KMA:

    """Main Functions"""
    def __init__(self, labeled, unlabeled, options, debug = False):
        self.ALPHA, self.LAMBDA, self.options = self.KMA(labeled, unlabeled, options, debug)

        #Set self.emb to be None
        self.emb = None

        #Create the symmetric Alpha
        self.sym_ALPHA = (self.ALPHA.T + self.ALPHA)/2

    def KMA(self, labeled, unlabeled, options, debug = False):

        #Convert to numpy array
        labeled = np.array(labeled)
        unlabeled = np.array(unlabeled)

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

        self.len_A = X_list[0].T.shape[0]
        self.len_B = X_list[1].T.shape[0]

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

        A = ((1 - options['mu']) * L + options['mu'] * Ls) + options.get('lambda', 1e-10) * np.eye(len(Ls)) #options.get('lambda', 1e-5) Modifiied number
        B = Ld

        # Kernel computation
        if options['kernelt'] == 'wan':
            Z = X_list[0]
            for X in X_list[1:]:
                Z = blkdiag(Z, X)

            
            KAK = Z @ A @ Z.T
            KBK = Z @ B @ Z.T

        elif options['kernelt'] == "lin":
            K = X_list[0].T @ X_list[0]

            for X in X_list[1:]:
                Ki = X.T @ X # Matrix multiplication
                K = blkdiag(K, Ki)  # Add to block diagonal matrix

            KAK = K @ A @ K
            KBK = K @ B @ K

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
            print(f"KAK MATRIX. Max: {np.max(KAK)}. Min: {np.min(KAK)}")
            imshow(KAK),show()

            input("Continue? Press Enter.")

            print(f"KBK MATRIX. Max: {np.max(KBK)}. Min: {np.min(KBK)}")
            imshow(KBK), show()
            input("Continue? Press Enter.")

        # Symmetrize KBK and KAK
        #KBK = (KBK + KBK.T) / 2
        #KAK = (KAK + KAK.T) / 2

        # Regularization to ensure KBK is positive semi-definite
        KBK += 1e-10 * np.eye(KBK.shape[0])

        # Solve the generalized eigenvalue problem
        M = np.dot(np.linalg.inv(KBK), KAK)
        M = (M + M.T) / 2  # Symmetrize the matrix product

        # Obtain eigenvalues and eigenvectors
        LAMBDA, ALPHA = np.linalg.eig(M)
        
        return ALPHA, LAMBDA, options

    """Plotting Functions and Comparisons"""
    def FOSCTTM(self, off_diagonal): 
        """
        FOSCTTM stands for average Fraction of Samples Closer Than the True Match.
        
        Lower scores indicate better alignment, as similar or corresponding points are mapped closer 
        to each other through the alignment process. If a method perfectly aligns all corresponding 
        points, the average FOSCTTM score would be 0. 

        :off_diagonal: should be either off-diagonal portion (that represents mapping from one domain to the other)
        of the block matrix. 
        """
        n1, n2 = np.shape(off_diagonal)
        if n1 != n2:
            raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

        dists = off_diagonal

        nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
        nn.fit(dists)

        _, kneighbors = nn.kneighbors(dists)

        return np.mean([np.where(kneighbors[i, :] == i)[0] / n1 for i in range(n1)])
    
    
    def cross_embedding_knn(self, embedding, labels, knn_args = {'n_neighbors': 4}):
        """Often abreviated as CE. 
        
        This trains a knn model using points only from one domain to predict the labels
        in the other domain. It returns the accuracy score. The closer to 1, the better."""

        (labels1, labels2) = labels

        n1 = len(labels1)

        knn = KNeighborsClassifier(**knn_args)
        knn.fit(embedding[:n1, :], labels1)

        return knn.score(embedding[n1:, :], labels2)

    def plot_emb(self, labels = None, n_comp = 2, show_lines = True, show_pred = False, show_legend = True, **kwargs): 
        """A useful visualization function to veiw the embedding.
        
        Arguments:
            :labels: should be a flattened list of the labels for points in domain A and then domain B. 
                If set to None, the cross embedding can not be calculated, and all points will be colored
                the same. 
            :n_comp: The amount of components or dimensions for the MDS function.
            :show_lines: should be a boolean value. If set to True, it will plot lines connecting the points 
                that correlate to the points in the other domain. It assumes a 1 to 1 correpondonce. 
            :show_anchors: should be a boolean value. If set to True, it will plot a black square on each point
                that is an anchor. 
            :**kwargs: additional key word arguments for sns.scatterplot function.
        """

        #Check to see if we already have created our embedding, else create the embedding.
        if type(self.emb) == type(None):
            #Convert to a MDS
            mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components= n_comp)
            self.emb = mds.fit_transform(self.sym_ALPHA)

        #Check to make sure we have labels
        if type(labels)!= type(None):
            #Seperate the labels into their respective domains
            first_labels = labels[:self.len_A]
            second_labels = labels[self.len_A:]

            #Calculate Cross Embedding Score
            try: #Will fail if the domain shapes aren't equal
                print(f"Cross Embedding: {self.cross_embedding_knn(self.emb, (first_labels, second_labels), knn_args = {'n_neighbors': 5})}")
            except:
                print("Can't calculate the Cross Embedding score")
        else:
            #Set all labels to be the same
            labels = np.ones(shape = (len(self.emb)))

        #Calculate FOSCTTM score
        try:    
            print(f"FOSCTTM: {self.FOSCTTM(self.sym_ALPHA[self.len_A:, :self.len_A])}") #This gets the off-diagonal part
        except: #This will run if the domains are different shapes
            print("Can't compute FOSCTTM with different domain shapes.")

        #Set the styles to show if a point comes from the first domain or the second domain
        styles = ['Domain A' if i < self.len_A else 'Domain B' for i in range(len(self.emb[:]))]

        #Create the figure
        plt.figure(figsize=(14, 8))

        #If show_pred is chosen, we want to show labels in Domain B as muted
        if show_pred:
            ax = sns.scatterplot(x = self.emb[self.len_A:, 0], y = self.emb[self.len_A:, 1], color = "grey", s=120, marker= "o", **kwargs)
            ax = sns.scatterplot(x = self.emb[:self.len_A, 0], y = self.emb[:self.len_A, 1], hue = Categorical(first_labels), s=120, marker= "^", **kwargs)
        else:
            #Now plot the points with correct lables
            ax = sns.scatterplot(x = self.emb[:, 0], y = self.emb[:, 1], style = styles, hue = Categorical(labels), s=120, markers= {"Domain A": "^", "Domain B" : "o"}, **kwargs)

        #Set the title and plot Legend
        ax.set_title("KMA", fontsize = 25)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        #Plot Legend
        if show_legend:
            plt.legend()

        #To plot line connections
        if show_lines:
            
            #Since this assumes 1 to 1 correpsondence, we must chech that the domains sizes are the same
            if self.len_A == self.len_B:
              for i in range(self.len_B):
                  ax.plot([self.emb[0 + i, 0], self.emb[self.len_A + i, 0]], [self.emb[0 + i, 1], self.emb[self.len_A + i, 1]], alpha = 0.65, color = 'lightgrey') #alpha = .5
            else:
               raise AssertionError("To show the lines, domain A and domain B must be the same size.")

        #Show plot
        plt.show()

        #Show the predicted points
        if show_pred and type(labels) != type(None):

            #Instantial model, fit on domain A, and predict for domain B
            knn_model = KNeighborsClassifier(n_neighbors=4)
            knn_model.fit(self.emb[:self.len_A, :], first_labels)
            second_pred = knn_model.predict(self.emb[self.len_A:, :])

            #Create the figure
            plt.figure(figsize=(14, 8))

            #Now plot the points
            ax = sns.scatterplot(x = self.emb[:, 0], y = self.emb[:, 1], style = styles, hue = Categorical(np.concatenate([first_labels, second_pred])), s=120, markers= {"Domain A": "^", "Domain B" : "o"}, **kwargs)

            #Set the title
            ax.set_title("Predicted Labels",  fontsize = 25)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.show()


