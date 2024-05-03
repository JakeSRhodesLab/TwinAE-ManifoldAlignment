#Diffusion Integration with Graphs

#Install the needed libraries
from scipy.spatial.distance import pdist, squareform
import graphtools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import seaborn as sns
from phate import PHATE
from vne import find_optimal_t
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

class DIG: #Diffusion Integration with Graphs
    def __init__(self, dataA, dataB, known_anchors, t = -1, knn = 5, link = "None"):
        """DataA is the first domain (or data set). DataB is the second domain (or data set). 
        
        Known Anchors should be in an array shape (n, 2), where n is the number of
        corresponding points and index [n, 0] is the node in dataA that corresponds to [n, 1]
        which is the node found in dataB
        
        t is the power to which we want to raise our diffusion matrix. If set to negative 1, it will find the optimal t
        
        Knn is the value of the amount of nearest neighbors we want. 
        
        Link determines if we want to apply Page Ranking or not. 'off-diagonal' means we only 
        want to apply the Page Ranking algorithm to the off-diagonal matricies, and 'full' mean we want to apply the page
        ranking algorithm across the entire block matrix."""

        #Scale data
        self.dataA = self.normalize_0_to_1(dataA)
        self.dataB = self.normalize_0_to_1(dataB)

        #Create graphs with graphtools
        self.graph_a = graphtools.Graph(self.dataA, knn = knn, knn_max = knn, decay = 40) #The Knn max stops additional connections
        self.graph_b  = graphtools.Graph(self.dataB, knn = knn, knn_max = knn, decay = 40)

        #Normalize the graphs
        self.kernalsA  = np.array(self.graph_a.K.toarray())
        self.kernalsB = np.array(self.graph_b.K.toarray())

        self.known_anchors = known_anchors

        #Connect the graphs
        self.graphAB = self.merge_graphs()
        
        #Get Similarity matrix and distance matricies
        self.similarity_matrix = self.get_pure_matricies(self.graphAB)

        #Get Diffusion Matrix
        self.sim_diffusion_matrix, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix, t, link = link)

        #Try doing diffusion the empy block way
        self.empty_block = self.get_zeros_and_ones_block()
        self.empty_diffused, __, __ = self.get_diffusion(self.empty_block, t, link = link) #NOTE: We currently do not store these projections

    """EVALUATION FUNCTIONS BELOW"""
    def FOSCTTM(self, Wxy): #Wxy should be just the parrallel matrix
        n1, n2 = np.shape(Wxy)
        if n1 != n2:
            raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

        dists = Wxy

        nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
        nn.fit(dists)

        _, kneighbors = nn.kneighbors(dists)

        return np.mean([np.where(kneighbors[i, :] == i)[0] / n1 for i in range(n1)])
    
    def cross_embedding_knn(self, embedding, Y, knn_args = {'n_neighbors': 1}):
        """Evaluation Metric that computes how many of the closest neighbors are correct"""
        (y1, y2) = Y

        n1, n2 = len(y1), len(y2)

        knn = KNeighborsClassifier(**knn_args)
        knn.fit(embedding[:n1, :], y1)

        return knn.score(embedding[n1:, :], y2)

    """HELPER FUNCTIONS BELOW"""
    def normalize_0_to_1(self, value):
        return (value - value.min()) / (value.max() - value.min())

    def apply_page_rank(self, matrix, alpha = 0.95):
        """
        Applies the PageRank modifications to the normalized matrix.

        Parameters:
        - matrix: The row-normalized adjacency matrix.
        - Alpha: the alpha value.

        Returns:
        - The modified matrix incorporating the damping factor and teleportation.
        """
        #Get the shape
        N, M = matrix.shape

        # Apply the damping factor and add the teleportation matrix
        return alpha * matrix + (1 - alpha) * np.ones((N, M)) / N

    def row_normalize_matrix(self, matrix):
        """This row normalizes the matrix """
        # Normalize the matrix so that each row sums to 1
        row_sums = matrix.sum(axis=1)
        return matrix / row_sums[:, np.newaxis]

    def get_pure_matricies(self, graph):
        """ Returns the similarity matrix"""

        matrix = graph.K.toarray()
        
        #Normalize the matrix
        matrix = self.normalize_0_to_1(matrix)

        return matrix
    
    def get_zeros_and_ones_block(self):
        """This creates off-diagonal blocks with known anchor points set to 1, and all other values set to 0. It returns a block matrix"""
        #Fill with zeros
        off_diagonal_block = np.zeros(shape=(self.len_A, self.len_B))

        #Set the anchors to 1
        off_diagonal_block[self.known_anchors[:, 0], self.known_anchors[:, 1]] = 1

        #Create block matrix
        block = np.block([[self.normalize_0_to_1(self.kernalsA), off_diagonal_block],
                         [off_diagonal_block.T, self.normalize_0_to_1(self.kernalsB)]])

        return block
    
    """THE PRIMARY FUNCTIONS"""
    def merge_graphs(self): #NOTE: This process takes a significantly longer with more KNN (O(N) complexity)
        """Creates a new graph from A and B using the known_anchors
        and by merging them together"""

        graphA = self.graph_a.to_igraph()
        graphB = self.graph_b.to_igraph()

        #This stores the length of graph A and B
        self.len_A = graphA.vcount()
        self.len_B = graphB.vcount()

        #Merge the two graphs together
        merged = graphA.disjoint_union(graphB) #Note: The distances between graphs may not be the same. It this is the case, would we want to scale the data first?

        #Change known_anchors to correspond between the new indexes
        self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

        #Now conenct the anchors together.
        for anchor in self.known_anchors: 
            neighborsA = tuple(set(graphA.neighbors(anchor[0], mode="out"))) #Anchor 0 applies to the graph A
            neighborsB = tuple(set(graphB.neighbors(anchor[1], mode="out")))

            #We add the edges first to a list so we can bulk add them later... NOTE: Since we are taking the weights from the kernal and not the graph object, it may be slightly different. It looks worse when we test the Projections, but the FOSCTTM score seems to be higher
            weights_to_add = self.kernalsB[neighborsB, np.repeat(anchor[1], len(neighborsB))]
            weights_to_add = np.append(weights_to_add, self.kernalsA[neighborsA, np.repeat(anchor[0], len(neighborsA))])

            #Bulk add the Edges
            edges_to_add = list(zip(np.full_like(neighborsB, anchor[0]), np.array(neighborsB) + self.len_A)) + list(zip(np.full_like(neighborsA, anchor[1]) + self.len_A, neighborsA)) #The self.len_A is to get it to correlate with the point in the domain B
            merged.add_edges(edges_to_add)
            
            #Add the weights
            merged.es[-len(weights_to_add):]["weight"] = weights_to_add

        #Now add the edges between anchors
        merged.add_edges(list(zip(self.known_anchors_adjusted[:, 0], self.known_anchors_adjusted[:, 1])))
        merged.es[-len(self.known_anchors_adjusted):]["weight"] = np.repeat(1, len(self.known_anchors_adjusted))

        #Convert back to graphtools
        merged_graphtools = graphtools.api.from_igraph(merged)

        return merged_graphtools

    def get_diffusion(self, matrix, t = -1, link = "None"): 
        """Returns the diffision matrix from the given matrix, to t steps. If t is -1, it will auto find the best one

        Also returns the projection matricies (the first one is the top right, and the second one is the bottom left)
        """
        #Find best T value
        if t == -1:
            t = find_optimal_t(matrix) #NOTE: Try checking the off diagonal with the entropy
            #print(f"Using optimal t value of {t}")

        # Normalize the matrix
        normalized_matrix = self.row_normalize_matrix(matrix)

        #Create small conections between all values 
        if link == "full":
            #print("Applying Page Ranking against the full matrix")
            normalized_matrix = self.apply_page_rank(normalized_matrix)
            normalized_matrix = self.row_normalize_matrix(normalized_matrix)
        elif link == "off-diagonal":
            #Get off-Diagonal blocks and apply the transformation
            #print("Applying Rage Ranking against the off-diagonal parts of the matrix")
            normalized_matrix[:self.len_A, self.len_A:] = self.apply_page_rank(normalized_matrix[:self.len_A, self.len_A:]) #Top right
            normalized_matrix[self.len_A:, :self.len_A] = self.apply_page_rank(normalized_matrix[self.len_A:, :self.len_A]) #Bottom left
            normalized_matrix = self.row_normalize_matrix(normalized_matrix)
            

        #Raise the normalized matrix to the t power
        diffusion_matrix = np.linalg.matrix_power(normalized_matrix, t)

        #Prepare the Projection Matricies by normalizing each domain by itself
        domainBA = diffusion_matrix[:self.len_A, self.len_A:] #Bottom Left
        domainAB = diffusion_matrix[self.len_A:, :self.len_A] #Top right
        domainBA = self.row_normalize_matrix(domainBA)
        domainAB = self.row_normalize_matrix(domainAB)
        
        #Squareform it :)
        diffused = (squareform(pdist((-np.log(0.00001+diffusion_matrix))))) #We can drop the -log and the 0.00001, but we seem to like it
    

        return diffused, domainBA, domainAB

    def predict_feature(self, predict = "A"):
        """Embedding should be the embedding wanted.
        
        predict should be which graph you want to use. 'A' for graph A. 
        'B' to predict graph B features. """

        if predict == "A":
            known_features = self.dataA
            projection_matrix = self.projectionBA #Bottom Left
        elif predict == "B":
            known_features = self.dataB
            projection_matrix = self.projectionAB #Top Right
        else:
            print("Please specify which features you want to predict. Graph 'A' or Graph 'B'")
            return None
        
        
        predicted_features = (projection_matrix[:, :, np.newaxis] * known_features[np.newaxis, :, :]).sum(axis = 1)
        
        return predicted_features
    
    def get_merged_data_set(self):
        """Adds the predicted features to the datasets with the missing features. 
        Returns a combined dataset that includes the predicted features"""

        #Add the predicted features to each data set
        full_data_A = np.hstack([self.predict_feature(predict = 'A'), self.dataB])
        full_data_B = np.hstack([self.dataA, self.predict_feature(predict = 'B')])

        #Combine the datasets
        completeData = np.vstack([full_data_A, full_data_B])

        return completeData
    
    """VISUALIZE AND TEST FUNCTIONS"""
    def plot_graphs(self):
        fig, axes = plt.subplots(2, 3, figsize = (13, 9))
        axes[0, 0].imshow(self.kernalsA)
        axes[0,0].set_title("Graph A Similarities")

        #Graph B
        axes[1, 0].imshow(self.empty_block)
        axes[1,0].set_title("Empty Block")

        #Similarity matrix
        axes[0,1].imshow(self.similarity_matrix)
        axes[0,1].set_title("Similarity Matrix")

        #Distance matrix
        axes[1,2].imshow(self.projectionAB)
        axes[1,2].set_title("Projection AB")

        #Diffusion matrix
        axes[0,2].imshow(self.sim_diffusion_matrix)
        axes[0,2].set_title("Similarities Diffusion Matrix")

        #Distance diffusion matrix
        axes[1,1].imshow(self.empty_diffused)
        axes[1,1].set_title("Empty diffused")

        plt.show()

    def plot_emb(self, labels, block, n_comp = 2, show_lines = True, show_anchors = True, use_phate = False, **kwargs): 
        #Convert to a MDS
        if use_phate:
            phate = PHATE(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components= n_comp)
            self.emb = phate.fit_transform(block)
        else:
            mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components= n_comp)
            self.emb = mds.fit_transform(block) #Later we should implent this to just be the block

        #Stress is a value of how well the emd did. Lower the better.
        print(f"Model Stress: {mds.stress_}")

        #Print the evaluation metrics as well
        first_labels = labels[:self.len_A]
        second_labels = labels[self.len_A:]
        try: #Will fail if the domains shapes aren't equal
            print(f"Cross Embedding: {self.cross_embedding_knn(self.emb, (first_labels, second_labels), knn_args = {'n_neighbors': 5})}")
        except:
            print("Can't calculate the Cross embedding")

        try:    
            print(f"FOSCTTM: {self.FOSCTTM(block[self.len_A:, :self.len_A])}") #This gets the off-diagonal part
        except: #This will run if the domains are different shapes
            print("Can't compute FOSCTTM with different domain shapes.")

        #Veiw the manifold. Those shown as Triangles are from GX
        styles = ['graph 1' if i < self.len_A else 'graph 2' for i in range(len(self.emb[:]))]
        plt.figure(figsize=(14, 8))

        #Now plot the points
        ax = sns.scatterplot(x = self.emb[:, 0], y = self.emb[:, 1], style = styles, hue = pd.Categorical(labels), s=80, markers= {"graph 1": "^", "graph 2" : "o"}, **kwargs)

        #To plot line connections
        if show_lines:
            for i in range(self.len_B):
                ax.plot([self.emb[0 + i, 0], self.emb[self.len_A + i, 0]], [self.emb[0 + i, 1], self.emb[self.len_A + i, 1]], color = 'lightgrey', alpha = .5)

        #Put black dots on the Anchors
        if show_anchors:
            ax.scatter(self.emb[self.known_anchors, 0], self.emb[self.known_anchors, 1], s = 10, color = 'black', marker="s")
        
        #Show plot
        plt.show()

    def plot_projection(self, labels, n_comp = 2, use_phate = False, use_original_data = False):
        """Plots the data in both Domain A and Domain B
        
        Matrix should be the union of the two graphs. """

        #Seperate domains from matrix
        domainA = self.similarity_matrix[:self.len_A, :self.len_A]
        domainB = self.similarity_matrix[self.len_A:, self.len_A:]

        #Create the Emb
        if use_phate:
            phate = PHATE(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components= n_comp)
            embA = phate.fit_transform(domainA)
            embB = phate.fit_transform(domainB)
        else: #The first section is using similarities and the second is without it being precomputed
            if use_original_data:
                mds = MDS(metric=True, random_state = 42, n_components= n_comp) #Note, we could also compute this using the original data
                embA = mds.fit_transform(self.dataA)
                embB = mds.fit_transform(self.dataB)
            else:
                mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components= n_comp) #Note, we could also compute this using the original data
                embA = mds.fit_transform(1 - domainA)
                embB = mds.fit_transform(1 - domainB)  



        #Multiply our embeddings across the projections
        projectedA = (embA[:, None, :] * self.projectionAB[:, :, None]).sum(axis=0)
        projectedB = (embB[:, None, :] * self.projectionBA[:, :, None]).sum(axis=0)

        #Add the projected points to each embedding
        embA = np.vstack((embA, projectedB)) #The domains may not be equal to the original
        embB = np.vstack((projectedA, embB))

        #Plot the two graphs
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

        #Create custom styles for each graph
        styles = ['Domain A' if i < self.len_A else 'Domain B' for i in range(len(embA[:]))]
        labelsA = np.concatenate([labels[:self.len_A], labels[:self.len_A]])
        axes[0].set_title("Domain A")
        sns.scatterplot(x = embA[:, 0], y = embA[:, 1], style = styles, hue = pd.Categorical(labelsA), ax = axes[0], s=80, markers= {"Domain A": "^", "Domain B" : "o"})

        styles = ['Domain A' if i < self.len_B else 'Domain B' for i in range(len(embB[:]))]
        labelsB = np.concatenate([labels[self.len_A:], labels[self.len_A:]])
        axes[1].set_title("Domain B")
        sns.scatterplot(x = embB[:, 0], y = embB[:, 1], style = styles, hue = pd.Categorical(labelsB), ax = axes[1], s=80, markers= {"Domain A": "^", "Domain B" : "o"})

        #Show plot
        plt.show()

    def plot_t_grid(self, block, rate = 3):
        """Returns the best T value according to its FOSCTTM score"""
        fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 16)) # Creates a grid of 2x5 for the subplots
        F_scores = np.array([])
        for i in range(1, 11):
            # Calculate the row and column index for the current subplot
            row = (i - 1) // 5
            col = (i - 1) % 5
            
            # Perform the diffusion operation
            a = i * rate
            diffused_array, projectionAB, projectionBA = self.get_diffusion(block, a)

            #Add Foscttm score
            F_scores = np.append(F_scores, self.FOSCTTM(diffused_array))
            
            # Plotting the diffused array
            ax = axes[row, col]
            ax.imshow(diffused_array)
            ax.set_title(f'T value {a}, FOSCTTM {(F_scores[i-1]):.4g}')
            ax.axis('off') # Hide the axes ticks

            #Plotting the associated Projections
            ax = axes[row+2, col]
            ax.imshow(projectionAB)
            ax.set_title(f'ProjectionAB: T value {a}')
            ax.axis('off')
            

        plt.tight_layout()

        print(f"The best T value is {(F_scores.argmin() +1) * rate} with a FOSCTTM of {(F_scores.min()):.4g}")


