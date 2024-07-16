#DIG (Diffusion Integration with Graphs)

#Import the needed libraries
import graphtools
import numpy as np
from pandas import Categorical
import seaborn as sns
from vne import find_optimal_t
from itertools import takewhile
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


class DIG: #Diffusion Integration with Graphs
    def __init__(self, t = -1, knn = 5, page_rank = "None", precompute = False, density_normalization = False, DTM = "log", verbose = 0, **kwargs):
        """
        Parameters:
            :t: the power to which we want to raise our diffusion matrix. If set to 
                negative 1 or any string, DIG will find the optimal t value.

            :KNN: should be an integer. Represents the amount of nearest neighbors to 
                construct the graphs.

            :page_rank: Determines if we want to apply Page Ranking or not. 'off-diagonal' means we only 
                want to apply the Page Ranking algorithm to the off-diagonal matricies, and 'full' 
                mean we want to apply the page ranking algorithm across the entire block matrix.

            :density_normalization: A boolean value. If set to true, it will apply a density
                normalization to the joined domains. 

            :DTM: Diffusion Transformation method. Can be set to "hellinger", "kl" or "log"
            :kwargs: Key word arguments for graphtools.Graph functions. 
         """


        #Store the needed information
        self.t = t
        self.knn = knn
        self.page_rank = page_rank
        self.normalize_density = density_normalization
        self.DTM = DTM.lower()
        self.precompute = precompute
        self.verbose = verbose
        self.kwargs = kwargs
    
    def fit(self, dataA, dataB, known_anchors):
        """
        Parameters:
            :DataA: the first domain (or data set). 
            :DataB: the second domain (or data set). 
            :Known Anchors: It should be an array shaped (n, 2) where n is the number of
                corresponding nodes. The first index should be the data point from DataA
                that corresponds to DataB
        """

        #Scale the data
        self.dataA = self.normalize_0_to_1(dataA)
        self.dataB = self.normalize_0_to_1(dataB)

        #Check to see if we want to create graphs with a precomputed Pdist algorithm or simply graphtools dijkstra'a algorithm
        if self.precompute:

            #Create kernals
            self.kernalsA = self.get_SGDM(dataA)
            self.kernalsB = self.get_SGDM(dataB)

            #Create Graphs using our precomputed kernals
            self.graph_a = graphtools.Graph(self.kernalsA, knn = self.knn, knn_max = self.knn, decay = 40, **self.kwargs)
            self.graph_b  = graphtools.Graph(self.kernalsB, knn = self.knn, knn_max = self.knn, decay = 40, **self.kwargs)

        else:
            #Create Graphs and allow it to use the normal data
            self.graph_a = graphtools.Graph(self.dataA, knn = self.knn, knn_max = self.knn, decay = 40, **self.kwargs)
            self.graph_b  = graphtools.Graph(self.dataB, knn = self.knn, knn_max = self.knn, decay = 40, **self.kwargs)

            #Get the Kernal Data from the graphs
            self.kernalsA  = np.array(self.graph_a.K.toarray())
            self.kernalsB = np.array(self.graph_b.K.toarray())

        self.known_anchors = known_anchors
            
        #This stores the length of the datasets A and B
        self.len_A = len(self.dataA)
        self.len_B = len(self.dataB)

        #Change known_anchors to correspond to off diagonal matricies
        self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

        #Connect the graphs
        self.graphAB = self.merge_graphs()
        
        #Get Similarity matrix and distance matricies
        self.similarity_matrix = self.get_similarity_matrix(self.graphAB)

        #Get Diffusion Matrix. int_diff_dist stands for the integrated diffusion distance.
        self.int_diff_dist, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix)

    """<><><><><><><><><><><><><><><><><><><><>     EVALUATION FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
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
    
    def partial_FOSCTTM(self, off_diagonal, anchors):
        """Follows the smae format as FOSCTTM.
        
        :off_diagonal: should be either off-diagonal portion (that represents mapping from one domain to the other)
        of the block matrix. 
        
        This calculates only a subset of points. It is intended to be used with hold-out anchors to help 
        us gauge whether new connections yielded in a better alignment."""

        n1, n2 = np.shape(off_diagonal)
        if n1 != n2:
            raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

        dists = off_diagonal

        nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
        nn.fit(dists)

        _, kneighbors = nn.kneighbors(dists)

        return np.mean([np.where(kneighbors[i[0], :] == i[1])[0] / n1 for i in anchors])
    
    def cross_embedding_knn(self, embedding, labels, knn_args = {'n_neighbors': 4}):
        """Often abreviated as CE. 
        
        This trains a knn model using points only from one domain to predict the labels
        in the other domain. It returns the accuracy score. The closer to 1, the better."""

        (labels1, labels2) = labels

        n1 = len(labels1)

        knn = KNeighborsClassifier(**knn_args)
        knn.fit(embedding[:n1, :], labels1)

        return knn.score(embedding[n1:, :], labels2)

    """<><><><><><><><><><><><><><><><><><><><>     HELPER FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
    def normalize_0_to_1(self, value):
        return (value - value.min()) / (value.max() - value.min())
    
    def get_SGDM(self, data):
        """SGDM - Same Graph Distance Matrix.
        This returns the normalized distances within the domain.
        
        Data should be the data of that makes up the domain."""
        
        #Get the distance measures for each data point and squareform it.
        data = squareform(pdist(data))

        #Normalize it and return the data
        return self.normalize_0_to_1(data)
    
    def row_normalize_matrix(self, matrix):
        """Returns a row normalized matrix"""

        #Get the sum for each row
        row_sums = matrix.sum(axis=1)

        #Prefrom the row-normalized division
        return matrix / row_sums[:, np.newaxis]

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

    def get_similarity_matrix(self, matrix):
        """Applies adjustments to get the similarity Matrix
        
        Returns the similarity matrix"""

        #Apply density normalization
        if self.normalize_density:
            matrix = self.density_normalized_kernel(matrix)
        
        #Normalize the matrix
        matrix = self.normalize_0_to_1(matrix)

        return matrix

    def kl_divergence_matrix(self, matrix):
        """
        Calculate the KL divergence matrix between rows of two matrices in a vectorized manner.

        Link to KL divergence formula and definition: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

        Parameters:
        matrix (numpy.ndarray): This should be the diffused matrix

        Returns:
        numpy.ndarray: Divergence matrix
        """

        # Ensure there are no zero values to avoid division by zero
        matrix = np.where(matrix == 0, 1e-10, matrix)

        #Normalize and do all the math to preform the KL divergence
        matrix = self.normalize_0_to_1(squareform(pdist(np.sum(matrix[:, np.newaxis, :] * np.log(matrix[:, np.newaxis, :] / matrix[np.newaxis, :, :]), axis=2))))

        #Return the block matrix!
        return matrix 
    
    def density_normalized_kernel(self, K):
        """
        Compute the density-normalized kernel matrix.

        Parameters:
        K (numpy.ndarray): The original kernel matrix (n x n).

        Returns:
        numpy.ndarray: The density-normalized kernel matrix (n x n).
        """

        # Compute the density estimates p by summing the values of each row
        p = np.sum(K, axis=1)
        
        # Ensure p is a column vector
        p = p.reshape(-1, 1)
        
        # Compute the outer product of the density estimates
        p_outer = np.sqrt(p @ p.T)
        
        # Compute the density-normalized kernel
        K_norm = K / p_outer
        
        return K_norm
    
    def hellinger_distance_matrix(self, matrix):
        """
        This compares each row to each other row in the matrix with the Hellinger
        algorithm -- determining similarities between distributions. 
        
        Parameters:
        matrix (numpy.ndarray): Matrix for the computation. Is expected to be the block.
        
        Returns:
        numpy.ndarray: Distance matrix.
        """

        #Create a single matrix by stacking the two blocks
        matrix = np.vstack([matrix[:self.len_A, :self.len_A], matrix[self.len_A:, self.len_A:]])

        #Reshape the maticies
        sqrt_matrix1 = np.sqrt(matrix[:, np.newaxis, :])
        sqrt_matrix2 = np.sqrt(matrix[np.newaxis, :, :])

        # Calculate the squared differences
        squared_diff = (sqrt_matrix1 - sqrt_matrix2) ** 2
        
        # Sum along the last axis to get the sum of squared differences
        sum_squared_diff = np.sum(squared_diff, axis=2)
        
        # Calculate the Hellinger distances
        distances = np.sqrt(sum_squared_diff) / np.sqrt(2)
        
        return distances

    def find_new_connections(self, pruned_connections = [], connection_limit = None, threshold = 0.2): 
        """A helper function that finds and returns a list of the closest connections and their associated wieghts after alignment.
            
        Parameters:
            :connection_limit: should be an integer. If set, the function will find no more than the connection amount specified. 
            :threshold: should be a float.
                The threshold determines how similar a point has to be to another to be kept as a connection. 
            :pruned_connections: should be a list formated like (n1, n2) where n1 is a point in Domain A, and n2 is a point in Domain B.
                The node connections in this list will not be considered for possible connections. 
            
        returns the possible connections"""

        #Keep Track of known-connections by creating a mask of everywhere we have a connection
        known_connections = self.similarity_matrix > 0

        if self.verbose > 0:
            print(f"Total number of Known_connections: {np.sum(known_connections)}")

        
        #This is made into an array to ensure the self.int_diff_dist is not changed
        array = np.array(self.int_diff_dist)

        #Set our Known-connections to inf values so they are not found and changed
        array[known_connections] = np.inf

        #Modify our array just to be the off-diagonal portion
        array = array[:self.len_A, self.len_A:]

        #Set the pruned_connections to be infinite as well
        array[pruned_connections] = np.inf

        #Set the connection_limit to be 1/3 of available connections if no limit was given
        if connection_limit == None:
            connection_limit = int((np.min(array.shape) - len(self.known_anchors)) / 3)

        """ This section below actually finds and then curates potential anchors """

        
        # Flatten the array
        array_flat = array.flatten()

        # Sort the array so we can find the smallest values
        smallest_indices = np.argsort(array_flat)

        # Select the indices of the first smallest values equal to the connection limit
        smallest_indices = smallest_indices[:connection_limit]

        # Convert the flattened indices to tuple coordinates (row, column)
        coordinates = [np.unravel_index(index, array.shape) for index in smallest_indices]

        #Add in coordinate values as the third index in the tuple (row, column, value)
        coordinates = [(int(coordinate[0]), int(coordinate[1]), array[coordinate[0], coordinate[1]]) for coordinate in coordinates]

        #Select only coordinates whose values are less than the given threshold
        coordinates = np.array(list(takewhile(lambda x: x[2] < threshold, coordinates)))

        return coordinates

    """THE PRIMARY FUNCTIONS"""
    def merge_graphs(self): #NOTE: This process takes a significantly longer with more KNN (O(N) complexity)
        """Creates a new graph (called graphAB) from graphs A and B using the known_anchors,
        adding an edge set with weight of 1 (as it is a similarity measure).
        
        Returns the kernal array of graphAB"""

        #convert Graphs to Igraphs
        graphA = self.graph_a.to_igraph()
        graphB = self.graph_b.to_igraph()

        #Merge the two graphs together in a disjoint way
        merged = graphA.disjoint_union(graphB)

        #For each anchor we want to find its neighbors (so we can connect those same edges to the anchor in the other domain).
        for anchor in self.known_anchors: 
            #Find the neighbors for each anchor
            neighborsA = tuple(set(graphA.neighbors(anchor[0], mode="out"))) #Anchor 0 applies to the graph A
            neighborsB = tuple(set(graphB.neighbors(anchor[1], mode="out"))) #Anchor 1 applies to the graph B

            #We add the edge weights first to a list so we can bulk add them later... NOTE: Since we are taking the weights from the kernal and not the graph object, it may be slightly different. It looks worse when we test the Projections, but the FOSCTTM score seems to be higher
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

        return merged_graphtools.K.toarray()
    
    def get_diffusion(self, matrix, return_projection = True): 
        """
        Returns the powered diffusion opperator from the given matrix.
        Also returns the projection matrix from domain A to B, and then the projection matrix from domain B to A. 
        """

        #Find best T value if t is set to auto
        if self.t == -1 or type(self.t) != int:
            self.t = find_optimal_t(matrix) 

            #If we found a T
            if self.verbose > 0:
                print(f"Using optimal t value of {self.t}")
                
        # Row normalize the matrix
        normalized_matrix = self.row_normalize_matrix(matrix)

        #Apply the page rank algorithm
        if self.page_rank == "full":
            if self.verbose > 2:
                print("Applying Page Ranking against the full matrix")
            
            normalized_matrix = self.apply_page_rank(normalized_matrix)
            normalized_matrix = self.row_normalize_matrix(normalized_matrix)
        
        #Get off-Diagonal blocks and apply the Page Rank transformation
        elif self.page_rank == "off-diagonal":
            if self.verbose > 2:
                print("Applying Rage Ranking against the off-diagonal parts of the matrix")

            normalized_matrix[:self.len_A, self.len_A:] = self.apply_page_rank(normalized_matrix[:self.len_A, self.len_A:]) #Top right
            normalized_matrix[self.len_A:, :self.len_A] = self.apply_page_rank(normalized_matrix[self.len_A:, :self.len_A]) #Bottom left
            normalized_matrix = self.row_normalize_matrix(normalized_matrix)
            

        #Raise the normalized matrix to the t power
        diffusion_matrix = np.linalg.matrix_power(normalized_matrix, self.t)

        if return_projection:
            #Prepare the Projection Matricies by normalizing each domain by itself
            domainAB = diffusion_matrix[:self.len_A, self.len_A:]#Top Right
            domainBA = diffusion_matrix[self.len_A:, :self.len_A] #Bottom Left
            domainAB = self.row_normalize_matrix(domainAB)
            domainBA = self.row_normalize_matrix(domainBA)
        
        #The Hellinger algorithm requires that the matricies have the same shape
        if self.DTM == "hellinger" and self.len_A == self.len_B:
            #Apply the hellinger process
            diffused = self.hellinger_distance_matrix(diffusion_matrix)

        elif self.DTM == "kl" and self.len_A == self.len_B:
            #Apply the hellinger process
            diffused = self.kl_divergence_matrix(diffusion_matrix)

        else:
            if (self.DTM == "hellinger" or self.DTM == "kl") and self.verbose > 0:
                print("Unable to compute hellinger or kl because datasets are not the same size.")

            #Squareform it :) --> TODO: Test the -np.log to see if that helps or not... we can see if we can use sqrt and nothing as well. :)
            diffused = (squareform(pdist((-np.log(0.00001+diffusion_matrix))))) #We can drop the -log and the 0.00001, but we seem to like it
    
            #Normalize the matrix
            diffused = self.normalize_0_to_1(diffused)
        
        if return_projection:
            return diffused, domainAB, domainBA
        else:
            return diffused

    def optimize_by_creating_connections(self, epochs = 3, threshold = "auto", connection_limit = "auto", hold_out_anchors = []):
        """
        In an interative process, it gets the potential anchors after alignment, and then recalculates the similarity matrix and 
        diffusion opperator. It then tests this new alignment, and if it is better, keeps the alignment.

        Returns True if a new alignment was made, otherwise it returns False
        
        Parameters:
            :connection_limit: should be an integer. If set, it will cap out the max amount of anchors found. 
                Best values to try: 1/10, 1/5, or 10x the length of the data, or None. 
            :threshold: should be a float. If auto, the algorithm will determine it. It can not be higher than the median value of the dataset.
                The threshold determines how similar a point has to be to another to be considered an anchor
            :hold_out_anchors: Should be in the same format as known_anchors. These are used to validate the new alignment. Can be given 
                anchors already used, but it preforms best if these are unseen anchors. 
            :pruned_connections: should be a list formated like Known Anchors. The node connections in this list will not be considered
                for possible connections. 
            :epochs: the number of iterations the cycle will go through. 
        """

        #Create value to return
        added_connections = False

        #Show the original connections
        if self.verbose > 1:
            print("<><><><><> Beggining Tests. Original Connections show below <><><><><>")
            plt.imshow(self.similarity_matrix)
            plt.show()

        
        #Set pruned_connections to equal hold_out_anchhor connections if they exist, empty otherwise
        if len(hold_out_anchors) > 0:

            #First add the hold_out_anchor connections
            pruned_connections = list(hold_out_anchors)

            #Create empty lists that will hold the anchor's neighbors (because these are also known connections)
            hold_neighbors_A = []
            hold_neighbors_B = []

            #Add in the the connections of each neighbor to each anchor
            for anchor_pair in hold_out_anchors:

                #Cache the data
                hold_neighbors_A += [(neighbor, anchor_pair[1]) for neighbor in set(self.graph_a.to_igraph().neighbors(anchor_pair[0], mode="out"))]
                hold_neighbors_B += [(anchor_pair[0], neighbor) for neighbor in set(self.graph_b.to_igraph().neighbors(anchor_pair[1], mode="out"))]

            #Add the connections
            pruned_connections += hold_neighbors_A
            pruned_connections += hold_neighbors_B
            
            #Convert to Numpy array for advanced indexing (for later use)
            hold_neighbors_A = np.array(hold_neighbors_A)
            hold_neighbors_B = np.array(hold_neighbors_B)
            
        else:
            pruned_connections = []

        if threshold == "auto":
            #Set the threshold to be the 10% limit of the connections
            threshold = np.sort(self.int_diff_dist.flatten())[:int(len(self.int_diff_dist.flatten()) * .1)][-1]

        if connection_limit == "auto":
            #Set the connection limit to be 10x the shape (while not always the best value, its consistently good and much faster due to the need of using less epochs)
            connection_limit = 10 * self.len_A

        #Get the current score of the alignment, by calculating the FOSCTTM scores that correlate with the hold_out_anchors
        current_score = np.mean([self.partial_FOSCTTM(self.int_diff_dist[self.len_A:, :self.len_A], hold_out_anchors), self.partial_FOSCTTM(self.int_diff_dist[:self.len_A, self.len_A:], hold_out_anchors)])

        #Find the Max value for new connections to be set to
        max_weight = np.median(self.similarity_matrix[self.similarity_matrix != 0])

        #Make sure we aren't finding values greater than the max_weight
        if threshold > max_weight:
            max_weight = threshold + 0.01
        
        if self.verbose > 0:
            print(f"Edges wont be set with similarity measure above: {max_weight}")

        #-----------------------------------------------------------------      Rebuild Class for each epoch        -----------------------------------------------------------------    
        for epoch in range(0, epochs):
            
            if self.verbose > 0:
                print(f"<><><><><><><><><><><><>    Starting Epoch {epoch}    <><><><><><><><><><><><><>")

            #Find predicted anchors
            new_connections = self.find_new_connections(pruned_connections, threshold = threshold, connection_limit = connection_limit)

            #If no new connections are found, quit the process
            if len(new_connections) < 1:
                if self.verbose > 0:
                    print("No new_connections. Exiting process")

                #Add in the known anchors and reset the known_anchors, similarity_matrix, and diffusion matrix
                if len(hold_out_anchors) > 0:

                    #Cached info 
                    adjusted_hold_neighbors_B = hold_neighbors_B + self.len_A

                    #Set the anchors
                    self.similarity_matrix[hold_out_anchors[:, 0], hold_out_anchors[:, 1] + self.len_A] = 1
                    self.similarity_matrix[hold_out_anchors[:, 0] + self.len_A, hold_out_anchors[:, 1]] = 1

                    #Set the connections values to the top right block
                    self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1] + self.len_A] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]
                    self.similarity_matrix[hold_neighbors_A[:, 0] + self.len_A, hold_neighbors_A[:, 1]] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]

                    #Set the connection values to the bottom left block
                    self.similarity_matrix[hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]
                    self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]

                    #Reset the Diffusion Matrix
                    self.int_diff_dist, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix)

                    #Add in the hold out anchors to the known_anchors
                    self.known_anchors += hold_out_anchors

                #Return True if we had found a new alignment, otherwise false
                return added_connections

            #Continue to show connections
            if self.verbose > 0:
                print(f"New connections found: {len(new_connections)}")

            #Copy Similarity matrix
            new_similarity_matrix = np.array(self.similarity_matrix) #We do this redudant conversion to an array to ensure we aren't copying over a reference

            #Add the new connections
            new_similarity_matrix[new_connections[:, 0].astype(int), (new_connections[:, 1] + self.len_A).astype(int)] = max_weight - new_connections[:, 2] #The max_weight minus is supposed to help go from distance to similarities
            new_similarity_matrix[(new_connections[:, 0] + self.len_A).astype(int) , new_connections[:, 1].astype(int)] = max_weight - new_connections[:, 2] #This is so we get the connections in the other off-diagonal block

            #Show the new connections
            if self.verbose > 1:
                plt.imshow(new_similarity_matrix)
                plt.show()

            #Get new Diffusion Matrix
            new_int_diff_dist = self.get_diffusion(new_similarity_matrix, return_projection = False)

            #Get the new alignment score
            new_score = np.mean([self.partial_FOSCTTM(new_int_diff_dist[self.len_A:, :self.len_A], hold_out_anchors), self.partial_FOSCTTM(new_int_diff_dist[:self.len_A, self.len_A:], hold_out_anchors)])

            #See if the extra connections helped
            if new_score < current_score or len(hold_out_anchors) < 1:

                if self.verbose > 0:
                    print(f"The new connections improved the alignment by {current_score - new_score}\n-----------     Keeping the new alignment. Continuing...    -----------\n")

                #Reset all the class variables. We don't worry about the calculating the projection matricies until the last epoch.
                self.similarity_matrix = new_similarity_matrix
                self.int_diff_dist = new_int_diff_dist

                #Reset the score
                current_score = new_score

                #Change Added connections to True
                added_connections = True

            else:
                if self.verbose > 0:
                    print(f"The new connections worsened the alignment by {new_score - current_score}\n-----------     Pruning the new connections. Continuing...    -----------\n")

                #Add the added connections to the the pruned_connections
                if len(pruned_connections) < 1:
                    pruned_connections = new_connections[:, :2].astype(int)
                else:
                    pruned_connections = np.concatenate([pruned_connections, new_connections[:, :2]]).astype(int)

        #On the final epoch, we can evaluate with the hold_out_anchors and then assign them as anchors. Also we need to ensure we calculate the projection matricies
        if epoch == epochs - 1:

            #Add in hold_out_anchors if applicable
            if len(hold_out_anchors) > 0:
                #Cached info 
                adjusted_hold_neighbors_B = hold_neighbors_B + self.len_A

                #Set the anchors
                self.similarity_matrix[hold_out_anchors[:, 0], hold_out_anchors[:, 1] + self.len_A] = 1
                self.similarity_matrix[hold_out_anchors[:, 0] + self.len_A, hold_out_anchors[:, 1]] = 1

                #Set the connections values to the top right block
                self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1] + self.len_A] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]
                self.similarity_matrix[hold_neighbors_A[:, 0] + self.len_A, hold_neighbors_A[:, 1]] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]

                #Set the connections values to the bottom left block
                self.similarity_matrix[hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]
                self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]

            #Recalculate diffusion matrix
            self.int_diff_dist, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix)

            #Show the final connections
            if self.verbose > 1:
                print("Added Hold Out Anchor Conections")
                plt.imshow(self.similarity_matrix)
                plt.show()
            
        #Process Finished
        if self.verbose > 0:
            print("<><><><><><><><><><<><><><><<> Epochs Finished <><><><><><><><><><><><><><><><><>")

        #Add in the hold out anchors to the known_anchors
        self.known_anchors += hold_out_anchors

        return added_connections

    """PREDICTING FEATURE FUNCTIONS"""
    def predict_feature(self, predict_with = "A"):
        """
        Predicts the the feature values from one domain to the other using the projection matricies. 

        Arguments:
        predict_with should be which graph data you want to use. 'A' for graph A and 'B' for graph B.
        
        Return the predicted features in an array
        """

        if predict_with == "A":
            known_features = self.dataA
            projection_matrix = self.projectionBA #Bottom Left
        elif predict_with == "B":
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
    def plot_heat_maps(self):
        """
        Plots and shows the heat maps for the similarity matrix, powered diffusion opperator,
        and projection matrix.
        """
        fig, axes = plt.subplots(1, 3, figsize = (13, 9))

        #Similarity matrix
        axes[0].imshow(self.similarity_matrix)
        axes[0].set_title("Similarity Matrix")

        #Projection AB
        axes[2].imshow(self.projectionAB)
        axes[2].set_title("Projection AB")

        #Diffusion matrix
        axes[1].imshow(self.int_diff_dist)
        axes[1].set_title("Integrated Diffusion Distance Matricies")

        plt.show()

    def plot_emb(self, labels = None, n_comp = 2, show_lines = True, show_anchors = True, show_pred = False, **kwargs): 
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

        #Convert to a MDS
        mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components= n_comp)
        self.emb = mds.fit_transform(self.int_diff_dist)

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
            print(f"FOSCTTM: {self.FOSCTTM(self.int_diff_dist[self.len_A:, :self.len_A])}") #This gets the off-diagonal part
        except: #This will run if the domains are different shapes
            print("Can't compute FOSCTTM with different domain shapes.")

        #Set the styles to show if a point comes from the first domain or the second domain
        styles = ['Domain A' if i < self.len_A else 'Domain B' for i in range(len(self.emb[:]))]

        #Create the figure
        plt.figure(figsize=(14, 8))

        #Now plot the points
        ax = sns.scatterplot(x = self.emb[:, 0], y = self.emb[:, 1], style = styles, hue = Categorical(labels), s=80, markers= {"Domain A": "^", "Domain B" : "o"}, **kwargs)

        #Set the title
        ax.set_title("DIG")

        #To plot line connections
        if show_lines:
            
            #Since this assumes 1 to 1 correpsondence, we must chech that the domains sizes are the same
            if self.len_A == self.len_B:
              for i in range(self.len_B):
                  ax.plot([self.emb[0 + i, 0], self.emb[self.len_A + i, 0]], [self.emb[0 + i, 1], self.emb[self.len_A + i, 1]], alpha = 0.65, color = 'lightgrey') #alpha = .5
            else:
               raise AssertionError("To show the lines, domain A and domain B must be the same size.")
             
        #Put black dots on the Anchors
        if show_anchors:
            
            #For each anchor set, plot lines between them
            for i in self.known_anchors_adjusted:
              ax.plot([self.emb[i[0], 0], self.emb[i[1], 0]], [self.emb[i[0], 1], self.emb[i[1], 1]], color = 'grey')
            
            #Create a new style guide so every other point is a triangle or circle
            styles2 = ['graph 1' if i % 2 == 0 else 'graph 2' for i in range(len(self.known_anchors)*2)]

            #Plot the black triangles or circles on the correct points
            sns.scatterplot(x = np.array(self.emb[self.known_anchors_adjusted, 0]).flatten(), y = np.array(self.emb[self.known_anchors_adjusted, 1]).flatten(), style = styles2, markers= {"graph 1": "^", "graph 2" : "o"}, s = 30, color = "black")
        
        #Show plot
        plt.show()

        #Show the predicted points
        if show_pred and type(labels) != type(None):

            #Instantial model, fit on domain A, and predict for domain B
            knn_model = KNeighborsClassifier(n_neighbors=4)
            knn_model.fit(self.emb[:self.len_A, :], first_labels)
            second_pred = knn_model.predict(self.emb[self.len_A:, :])

            #Fit on domain A, and predict for domain B
            knn_model.fit(self.emb[self.len_A:, :], second_labels)
            first_pred = knn_model.predict(self.emb[:self.len_A, :])
            
            #Create the figure
            plt.figure(figsize=(14, 8))

            #Now plot the points
            ax = sns.scatterplot(x = self.emb[:, 0], y = self.emb[:, 1], style = styles, hue = Categorical(np.concatenate([first_pred, second_pred])), s=80, markers= {"Domain A": "^", "Domain B" : "o"}, **kwargs)

            #Set the title
            ax.set_title("Predicted Labels")

            plt.show()

    def plot_t_grid(self, rate = 3):
        """Plots the powered diffusion operator many times each with a different t value. Also plots
        the associated projection matrix. 
        
        Arguments:
            :rate: the value by which to increment t for each iteration.
            
        It has no return.
        """

        #Store the original t value
        t_str = str(self.t)
        
        #Create the figure
        fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 16))

        #Create an empty list to store the FOSCTTM scores
        F_scores = np.array([])

        for i in range(1, 11):
            # Calculate the row and column index for the current subplot
            row = (i - 1) // 5
            col = (i - 1) % 5
            
            # Perform the diffusion operation
            self.t = i * rate
            diffused_array, projectionAB, projectionBA = self.get_diffusion(self.similarity_matrix)

            #Calculate FOSCTTM score
            F_scores = np.append(F_scores, self.FOSCTTM(diffused_array))
            
            # Plotting the diffused array
            ax = axes[row, col]
            ax.imshow(diffused_array)
            ax.set_title(f'T value {self.t}, FOSCTTM {(F_scores[i-1]):.4g}')
            ax.axis('off')

            #Plotting the associated Projections
            ax = axes[row+2, col]
            ax.imshow(projectionAB)
            ax.set_title(f'ProjectionAB: T value {self.t}')
            ax.axis('off')
            
        #Show the plot
        plt.tight_layout()
        plt.show()

        #Restore t value
        self.t = int(t_str)

        print(f"The best T value is {(F_scores.argmin() +1) * rate} with a FOSCTTM of {(F_scores.min()):.4g}")
