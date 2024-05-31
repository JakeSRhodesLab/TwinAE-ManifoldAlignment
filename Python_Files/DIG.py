#Diffusion Integration with 

"""Ideas for upgrades
1. Auto recognition for point correspondence -> Potential errors (it could potentially be harmful if there isn't a 1 to 1 correspondence). Additional question: how helpful is it to have additional anchors?"""

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
    def __init__(self, dataA, dataB, known_anchors, t = -1, knn = 5, link = "None", verbose = 0):
        """DataA is the first domain (or data set). DataB is the second domain (or data set). 
        
        Known Anchors should be in an array shape (n, 2), where n is the number of
        corresponding points and index [n, 0] is the node in dataA that corresponds to [n, 1]
        which is the node found in dataB
        
        t is the power to which we want to raise our diffusion matrix. If set to negative 1, it will find the optimal t
        
        Knn is the value of the amount of nearest neighbors we want. 
        
        Link determines if we want to apply Page Ranking or not. 'off-diagonal' means we only 
        want to apply the Page Ranking algorithm to the off-diagonal matricies, and 'full' mean we want to apply the page
        ranking algorithm across the entire block matrix."""

        #For Need to know stuff 
        self.verbose = verbose

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
        """self.empty_block = self.get_zeros_and_ones_block()
        self.empty_diffused, __, __ = self.get_diffusion(self.empty_block, t, link = link) #NOTE: We currently do not store these projections"""

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
    
    def cross_embedding_knn(self, embedding, Y, knn_args = {'n_neighbors': 4}):
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
    
    def _find_possible_anchors(self, anchor_limit = None, threshold = "auto", hold_out_anchors = []): #TODO: Make it so each point can only have one correspondence
        """A helper function that finds and returns a list of possible anchors after alignment.
        
        Parameters:
        :anchor_limit: should be an integer. If set, it will cap out the max amount of anchors found.
        :threshold: should be a float. If auto, the algorithm will determine it.
        The threshold determines how similar a point has to be to another to be considered an anchor
        :hold_out_anchors: Only matters if Threshold is set to auto. These anchors are used as a test to validate the Threshold.
        They should be in the same format as the Known Anchors.
        
        returns possible anchors plus known anchors in a single list"""

        #Calculate the threshold
        if threshold == "auto":

            #Change Type so that we can convert to set 
            known_anchors_as_tuples = (tuple(arr) for arr in self.known_anchors)
            hold_out_anchors = [tuple(arr) for arr in hold_out_anchors]

            # Convert the list of tuples to a set for fast look ups
            set1 = set(known_anchors_as_tuples)

            #Remove indicies that are already known anchors
            hold_out_anchors[:] = [tup for tup in hold_out_anchors if tup not in set1]

            #Check to make sure we have Hold out anchors
            if len(hold_out_anchors) < 1:
                print("ERROR: No calculation preformed. Please provide hold_out_anchors and ensure they aren't known anchors already.")
                return []
            elif len(hold_out_anchors) < 2:
                #Since there is only one element, we set the threshold to be equal to its max plus a tiny bit
                threshold = self.sim_diffusion_matrix[hold_out_anchors[0][0], hold_out_anchors[0][1] + self.len_A]

            else:
                #Adjust the Hold_out_anchors to map in the merged graphs
                hold_out_anchors = np.array(hold_out_anchors)
                hold_out_anchors = np.vstack([hold_out_anchors.T[0], hold_out_anchors.T[1] + self.len_A]).T

                #Determine the average distance of the hold out anchors
                average_threshold = np.mean(self.sim_diffusion_matrix[hold_out_anchors[0], hold_out_anchors[1]]) #NOTE: we might have to adjust this value. 
                _65_percent_interval = np.std(self.sim_diffusion_matrix[hold_out_anchors[0], hold_out_anchors[1]])

                threshold = average_threshold + _65_percent_interval

        # Create a boolean mask where the distance is less than the threshold
        mask = self.sim_diffusion_matrix[:self.len_A, self.len_A:] < threshold

        # Use np.where to get the indices of the points that satisfy the condition
        possible_anchors = np.where(mask)

        #Transpose it
        possible_anchors = np.array(possible_anchors).T

        #For each domain, remove the known anchors
        possible_anchors = np.array([pair for pair in possible_anchors if pair[0] not in self.known_anchors[:, 0] and pair[1] not in self.known_anchors[:, 1]])
    
        #Since the data is injective, remove all duplicate values, choosing the smallest
        unique_possible_anchors = np.unique(possible_anchors[:, 0])
        possible_anchors = np.argmin(self.sim_diffusion_matrix[:self.len_A, self.len_A:][unique_possible_anchors], axis = 0)
        possbile_anchors = self.sim_diffusion_matrix[:self.len_A, self.len_A:][unique_possible_anchors, possible_anchors]

        #Transpose it
        possible_anchors = np.array(possible_anchors).T

        #Check to see if any anchors have been found
        if len(possible_anchors) < 1:
            print("ERROR: No Possible anchors found. Try increasing the threshold")
            return []

        #Apply the anchor Limit
        if type(anchor_limit) == int:
            # Extract the similarity values that are less than 0.1
            dist_values = self.sim_diffusion_matrix[:self.len_A, self.len_A:][mask]

            # Combine indices and values into a list of tuples
            indexed_values = list(zip(possible_anchors[:, 0], possible_anchors[:, 1], dist_values))

            # Sort the list of tuples by the similarity values (third element in the tuples)
            indexed_values = sorted(indexed_values, key=lambda x: x[2])

            # Print the indices and values of the first anchor Limit smallest similarities
            if self.verbose > 0:
                for i, (row, col, value) in enumerate(indexed_values[:anchor_limit]):
                    print(f"{i+1}: Index ({row}, {col}) - Similarity: {value}")
            
            # Select the first anchor_limit smallest values (or all if there are less than anchor_limit)
            possible_anchors = np.array(indexed_values)[:anchor_limit, 0:2].astype(int)

        #Add the anchors to the known anchors
        possible_anchors = np.concatenate((self.known_anchors, possible_anchors), axis = 0)

        return possible_anchors



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
        
        #Squareform it :) --> TODO: Test the -np.log to see if that helps or not... we can see if we can use sqrt and nothing as well. :)
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
    
    def recreate_with_new_anchors(self, epochs = 3, **find_possible_anchors_kwargs):
        """Finds potential anchors after alignment, and then recalculates the entire alignment with the new anchor points for each epoch. 
        
        Parameters:
        :anchor_limit: should be an integer. If fixed, it will determine the max anchors the algorithm will find.
        :epochs: the number of iterations the cycle will go through. 
        """

        #Reset the known anchors
        self.known_anchors = self._find_possible_anchors(**find_possible_anchors_kwargs)


        #On the final epoch, we can evaluate with the hold_out_anchors and then assign them as anchors. 
        pass
    """VISUALIZE AND TEST FUNCTIONS"""
    def plot_graphs(self):
        fig, axes = plt.subplots(2, 3, figsize = (13, 9))
        axes[0, 0].imshow(self.kernalsA)
        axes[0,0].set_title("Graph A Similarities")

        """ This is if we wanted the Empty Block
        #Graph B
        axes[1, 0].imshow(self.empty_block)
        axes[1,0].set_title("Empty Block")
        """

        #Similarity matrix
        axes[0,1].imshow(self.similarity_matrix)
        axes[0,1].set_title("Similarity Matrix")

        #Distance matrix
        axes[1,2].imshow(self.projectionAB)
        axes[1,2].set_title("Projection AB")

        #Diffusion matrix
        axes[0,2].imshow(self.sim_diffusion_matrix)
        axes[0,2].set_title("Similarities Diffusion Matrix")

        """This is if we wanted the Empty Block
        #Distance diffusion matrix
        axes[1,1].imshow(self.empty_diffused)
        axes[1,1].set_title("Empty diffused")
        """

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


