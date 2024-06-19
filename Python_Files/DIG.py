#Diffusion Integration with Graphs

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
from itertools import takewhile

class DIG: #Diffusion Integration with Graphs
    def __init__(self, dataA, dataB, known_anchors, t = -1, knn = 5, link = "None", density_normalization = False, hellinger = False, verbose = 0):
        """
        Parameters:
            :DataA: the first domain (or data set). 
            :DataB: the second domain (or data set). 
            :Known Anchors: It should be an array shaped (n, 2) where n is the number of
                corresponding nodes. The first index should be the data point from DataA
                that corresponds to DataB
            :t: the power to which we want to raise our diffusion matrix. If set to 
                negative 1, DIG will find the optimal t value.
            :KNN: should be an integer. Represents the amount of nearest neighbors to 
                construct the graphs.
            :link: Determines if we want to apply Page Ranking or not. 'off-diagonal' means we only 
                want to apply the Page Ranking algorithm to the off-diagonal matricies, and 'full' 
                mean we want to apply the page ranking algorithm across the entire block matrix.
            :density_normalization: A boolean value. If set to true, it will apply a density
                normalization to the joined domains. 

            :merge: If merge is set to True, it will use graphs, otherwise it will use kernals    
            """


        #For Need to know stuff 
        self.verbose = verbose
        self.t = t
        self.link = link
        self.normalize_density = density_normalization
        self.hellinger = hellinger

        #Scale data
        self.dataA = self.normalize_0_to_1(dataA)
        self.dataB = self.normalize_0_to_1(dataB)

        #Create graphs with graphtools
        self.graph_a = graphtools.Graph(self.dataA, knn = knn, knn_max = knn, decay = 40) #The Knn max stops additional connections
        self.graph_b  = graphtools.Graph(self.dataB, knn = knn, knn_max = knn, decay = 40)

        #Create the kernals
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
        self.similarity_matrix = self.get_pure_matricies(self.graphAB)

        #Get Diffusion Matrix
        self.sim_diffusion_matrix, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix, self.t, link = self.link)

        # NOTE: The above line returns the order projectionAB then projection BA, but get_diffusion returns BA then AB
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
    
    def partial_FOSCTTM(self, Wxy, anchors): #Wxy should be just the parrallel matrix
        """This uses only the provided known connections"""

        n1, n2 = np.shape(Wxy)
        if n1 != n2:
            raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

        dists = Wxy

        nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
        nn.fit(dists)

        _, kneighbors = nn.kneighbors(dists)

        return np.mean([np.where(kneighbors[i[0], :] == i[1])[0] / n1 for i in anchors])
    
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

    def get_pure_matricies(self, matrix):
        """ Returns the similarity matrix"""

        #Apply density normalization
        if self.normalize_density:
            matrix = self.density_normalized_kernel(matrix)
        
        #Normalize the matrix
        matrix = self.normalize_0_to_1(matrix)

        return matrix

    import numpy as np

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

        # Calculate the square roots of the matrices
        sqrt_matrix1 = np.sqrt(matrix[:, np.newaxis, :])
        sqrt_matrix2 = np.sqrt(matrix[np.newaxis, :, :])
        
        # Calculate the squared differences
        squared_diff = (sqrt_matrix1 - sqrt_matrix2) ** 2
        
        # Sum along the last axis to get the sum of squared differences
        sum_squared_diff = np.sum(squared_diff, axis=2)
        
        # Calculate the Hellinger distances
        distances = np.sqrt(sum_squared_diff) / np.sqrt(2)
        
        return distances

    def _find_new_connections(self, pruned_connections = [], connection_limit = None, threshold = 0.2): 
        """A helper function that finds and returns a list of possible anchors and their associated wieghts after alignment.
            
        Parameters:
            :connection_limit: should be an integer. If set, it will cap out the max amount of anchors found.
            :threshold: should be a float.
                The threshold determines how similar a point has to be to another to be considered an anchor
            :pruned_connections: should be a list formated like Known Anchors. The node connections in this list will not be considered
                for possible connections. 
            
        returns possible anchors plus known anchors in a single list"""

        #Keep Track of known-connections 
        known_connections = self.similarity_matrix > 0 #Creates a mask of everywhere we have a connection

        if self.verbose > 0:
            print(f"Total number of Known_connections: {np.sum(known_connections)}")

        #Set our Known-connections to inf values so they are not found and changed
        array = np.array(self.sim_diffusion_matrix) #This is made into an array to ensure we aren't passing by reference
        array[known_connections] = np.inf

        #Modify our array just to be the off-diagonal 
        array = array[:self.len_A, self.len_A:]

        #Add in our pruned connections
        array[pruned_connections] = np.inf

        #Set anchor limit to 1/3 of the unknown data points
        if connection_limit == None:
            #connection_limit = int((np.min(array.shape) - len(self.known_anchors)) / 3)
            connection_limit = int(array.shape[0] * array.shape[1])

        """ This section actually finds and then curates potential anchors """

        
        # Flatten the array
        array_flat = array.flatten()

        # Get the indices that would sort the array
        smallest_indices = np.argsort(array_flat)

        # Select the indices of the first 5 smallest values
        smallest_indices = smallest_indices[:connection_limit]

        # Convert the flattened indices to tuple coordinates (row, column)
        coordinates = [np.unravel_index(index, array.shape) for index in smallest_indices]

        #Add in coordinate values as the third index in the tuple
        coordinates = [(int(coordinate[0]), int(coordinate[1]), array[coordinate[0], coordinate[1]]) for coordinate in coordinates]

        #Apply the Threshold
        coordinates = np.array(list(takewhile(lambda x: x[2] < threshold, coordinates)))

        return coordinates

    """THE PRIMARY FUNCTIONS"""
    def merge_graphs(self): #NOTE: This process takes a significantly longer with more KNN (O(N) complexity)
        """Creates a new graph from A and B using the known_anchors
        and by merging them together"""

        graphA = self.graph_a.to_igraph()
        graphB = self.graph_b.to_igraph()

        #Merge the two graphs together
        merged = graphA.disjoint_union(graphB) #Note: The distances between graphs may not be the same. It this is the case, would we want to scale the data first?

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

        return merged_graphtools.K.toarray()
    
    def get_diffusion(self, matrix, t = -1, link = "None"): 
        """Returns the diffision matrix from the given matrix, to t steps. If t is -1, it will auto find the best one

        Also returns the projection matricies (the first one is the top right, and the second one is the bottom left)
        """
        #Find best T value
        if t == -1:
            t = find_optimal_t(matrix) #NOTE: Try checking the off diagonal with the entropy
            #print(f"Using optimal t value of {t}")


        #TODO -- Maybe add bit about the kernal densities (Might not be right spot)
        
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
        domainAB = diffusion_matrix[:self.len_A, self.len_A:]#Top Right
        domainBA = diffusion_matrix[self.len_A:, :self.len_A] #Bottom Left
        domainAB = self.row_normalize_matrix(domainAB)
        domainBA = self.row_normalize_matrix(domainBA)
        
        if self.hellinger:
            #Apply the hellinger process
            diffused = self.hellinger_distance_matrix(diffusion_matrix)
        else:
            #Squareform it :) --> TODO: Test the -np.log to see if that helps or not... we can see if we can use sqrt and nothing as well. :)
            diffused = (squareform(pdist((-np.log(0.00001+diffusion_matrix))))) #We can drop the -log and the 0.00001, but we seem to like it
    
            #Normalize the matrix
            diffused = self.normalize_0_to_1(diffused)

        return diffused, domainAB, domainBA

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

    def optimize_by_creating_connections(self, epochs = 3, threshold = "auto", connection_limit = "auto", hold_out_anchors = []):
        """Finds potential anchors after alignment, and then recalculates the entire alignment with the new anchor points for each epoch. 
        
        Parameters:
            :connection_limit: should be an integer. If set, it will cap out the max amount of anchors found. 
                Best values to try: 1/5 of the length of data, 1/10 length of the data, 10x length of data, or None. 
            :threshold: should be a float. If auto, the algorithm will determine it. It can not be higher than the median of the dataset.
                The threshold determines how similar a point has to be to another to be considered an anchor
            :hold_out_anchors: Only matters if Threshold is set to auto. These anchors are used as a test to validate the Threshold.
                They should be in the same format as the Known Anchors.
            :pruned_connections: should be a list formated like Known Anchors. The node connections in this list will not be considered
                for possible connections. 
            :epochs: the number of iterations the cycle will go through. 
        """

        #Show the original connections
        if self.verbose > 1:
            print("<><><><><> Beggining Tests. Original Connections show below <><><><><>")
            plt.imshow(self.similarity_matrix)
            plt.show()

        
        #Set pruned_connections to equal hold_out_anchhor connections if they exist, empty otherwise
        if len(hold_out_anchors) > 0:
            #Firt add the anchor connections (We do this first to A, as it will later be added to pruned anchors. This helps us later when adding the known values in at the end)
            pruned_connections = list(hold_out_anchors)

            hold_neighbors_A = []
            hold_neighbors_B = []

            #Add in the the connections of each neighbor to each anchor
            for anchor_pair in hold_out_anchors: #TODO: Vectorize this somehow?

                #Cache the data
                hold_neighbors_A += [(neighbor, anchor_pair[1]) for neighbor in set(self.graph_a.to_igraph().neighbors(anchor_pair[0], mode="out"))]
                hold_neighbors_B += [(anchor_pair[0], neighbor) for neighbor in set(self.graph_b.to_igraph().neighbors(anchor_pair[1], mode="out"))]

                #Add the connections
                pruned_connections += hold_neighbors_A
                pruned_connections += hold_neighbors_B
            
            #Convert to Numpy array for advanced indexing
            hold_neighbors_A = np.array(hold_neighbors_A)
            hold_neighbors_B = np.array(hold_neighbors_B)
            
        else:
            pruned_connections = []

        if threshold == "auto":
            #Set the threshold to be the 10% limit of the connections
            threshold = np.sort(self.sim_diffusion_matrix.flatten())[:int(len(self.sim_diffusion_matrix.flatten()) * .1)][-1]

        if connection_limit == "auto":
            #Set the connection limit to be 10x the shape (just because its usualy good and fast)
            connection_limit = 10 * self.len_A

        #Create an empty Numpy array
        pruned_connections = np.array([]).astype(int)

        #Get the current score of the alignment
        current_score = np.mean([self.partial_FOSCTTM(self.sim_diffusion_matrix[self.len_A:, :self.len_A], hold_out_anchors), self.partial_FOSCTTM(self.sim_diffusion_matrix[:self.len_A, self.len_A:], hold_out_anchors)])

        #Find the Max value for new connections to be set too
        second_max = np.median(self.similarity_matrix[self.similarity_matrix != 0])
        
        if self.verbose > 0:
            print(f"Second max: {second_max}")
        
        #Rebuild Class for each epoch
        for epoch in range(0, epochs):
            
            if self.verbose > 0:
                print(f"<><><><><><><><><><><><>    Starting Epoch {epoch}    <><><><><><><><><><><><><>")

            #Find predicted anchors
            new_connections = self._find_new_connections(pruned_connections, threshold = threshold, connection_limit = connection_limit)

            if len(new_connections) < 1:
                if self.verbose > 0:
                    print("No new_connections. Exiting process")

                #Add in the known anchors and reset the known_anchors and other init variables
                if len(hold_out_anchors) > 0:
                    #Cached info 
                    adjusted_hold_neighbors_B = hold_neighbors_B + self.len_A

                    #Set the anchors
                    self.similarity_matrix[hold_out_anchors[:, 0], hold_out_anchors[:, 1] + self.len_A] = 1
                    self.similarity_matrix[hold_out_anchors[:, 0] + self.len_A, hold_out_anchors[:, 1]] = 1

                    #Set the other connections (taking values from the top left)
                    self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1] + self.len_A] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]
                    self.similarity_matrix[hold_neighbors_A[:, 0] + self.len_A, hold_neighbors_A[:, 1]] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]

                    #Take connection values from the bottom right 
                    self.similarity_matrix[hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]
                    self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]

                    #Get Diffusion Matrix
                    self.sim_diffusion_matrix, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix, self.t, link = self.link)
                

                #Return false to signify we didn't go through all the tests
                return False

            #Continue to show connections
            if self.verbose > 0:
                print(f"New connections found: {len(new_connections)}")

            #Copy Similarity matrix
            new_similarity_matrix = np.array(self.similarity_matrix) #We do this redudant conversion to ensure we aren't copying over a reference

            #Add the new connections
            new_similarity_matrix[new_connections[:, 0].astype(int), (new_connections[:, 1] + self.len_A).astype(int)] = second_max - new_connections[:, 2] #The max_value minus is supposed to help go from distance to similarities
            new_similarity_matrix[(new_connections[:, 0] + self.len_A).astype(int) , new_connections[:, 1].astype(int)] = second_max - new_connections[:, 2] #This is so we get the connections in the other off-diagonal block

            #Show the new connections
            if self.verbose > 1:
                plt.imshow(new_similarity_matrix)
                plt.show()
            
            #Get new Diffusion Matrix
            new_sim_diffusion_matrix, new_projectionAB, new_projectionBA = self.get_diffusion(new_similarity_matrix, self.t, link = self.link)

            #Get the new score
            new_score = np.mean([self.partial_FOSCTTM(new_sim_diffusion_matrix[self.len_A:, :self.len_A], hold_out_anchors), self.partial_FOSCTTM(new_sim_diffusion_matrix[:self.len_A, self.len_A:], hold_out_anchors)])

            #See if the extra connections helped
            if new_score < current_score or len(hold_out_anchors) < 1:
                if self.verbose > 0:
                    print(f"The new connections improved the alignment by {current_score - new_score}\n-----------     Keeping the new alignment. Continuing...    -----------\n")

                #Reset all the class variables
                self.similarity_matrix = new_similarity_matrix
                self.sim_diffusion_matrix = new_sim_diffusion_matrix
                self.projectionAB = new_projectionAB
                self.projectionBA = new_projectionBA

                #Reset the score
                current_score = new_score

            else:
                if self.verbose > 0:
                    print(f"The new connections worsened the alignment by {new_score - current_score}\n-----------     Pruning the new connections. Continuing...    -----------\n")

                #Add the added connections to the the pruned_connections
                if len(pruned_connections) < 1:
                    pruned_connections = new_connections[:, :2].astype(int)
                else:
                    pruned_connections = np.concatenate([pruned_connections, new_connections[:, :2]]).astype(int)

        #On the final epoch, we can evaluate with the hold_out_anchors and then assign them as anchors. 
        if epoch == epochs - 1 and len(hold_out_anchors) > 0:
            #Cached info 
            adjusted_hold_neighbors_B = hold_neighbors_B + self.len_A

            #Set the anchors
            self.similarity_matrix[hold_out_anchors[:, 0], hold_out_anchors[:, 1] + self.len_A] = 1
            self.similarity_matrix[hold_out_anchors[:, 0] + self.len_A, hold_out_anchors[:, 1]] = 1

            #Set the other connections (taking values from the top left)
            self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1] + self.len_A] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]
            self.similarity_matrix[hold_neighbors_A[:, 0] + self.len_A, hold_neighbors_A[:, 1]] = self.similarity_matrix[hold_neighbors_A[:, 0], hold_neighbors_A[:, 1]]

            #Take connection values from the bottom right 
            self.similarity_matrix[hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]
            self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], hold_neighbors_B[:, 1]] = self.similarity_matrix[adjusted_hold_neighbors_B[:, 0], adjusted_hold_neighbors_B[:, 1]]

            #Recalculate diffusion matrix
            self.sim_diffusion_matrix, self.projectionAB, self.projectionBA = self.get_diffusion(self.similarity_matrix, self.t, link = self.link)

            #Show the final connections
            if self.verbose > 1:
                print("Added Hold Out Anchor Conections")
                plt.imshow(self.similarity_matrix)
                plt.show()
            
        #Process Finished
        if self.verbose > 0:
            print("<><><><><><><><><><<><><><><<> Epochs Finished <><><><><><><><><><><><><><><><><>")

        return True

    """VISUALIZE AND TEST FUNCTIONS"""
    def plot_graphs(self):
        fig, axes = plt.subplots(1, 3, figsize = (13, 9))

        #Similarity matrix
        axes[0].imshow(self.similarity_matrix)
        axes[0].set_title("Similarity Matrix")

        #Projection AB
        axes[2].imshow(self.projectionAB)
        axes[2].set_title("Projection AB")

        #Diffusion matrix
        axes[1].imshow(self.sim_diffusion_matrix)
        axes[1].set_title("Similarities Diffusion Matrix")

        plt.show()

    def plot_emb(self, labels = None, n_comp = 2, show_lines = True, show_anchors = True, **kwargs): 
        #Convert to a MDS
        mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components= n_comp)
        self.emb = mds.fit_transform(self.sim_diffusion_matrix) #Later we should implent this to just be the block

        #Stress is a value of how well the emd did. Lower the better.
        print(f"Model Stress: {mds.stress_}")

        if labels != None:
            #Print the evaluation metrics as well
            first_labels = labels[:self.len_A]
            second_labels = labels[self.len_A:]
            try: #Will fail if the domains shapes aren't equal
                print(f"Cross Embedding: {self.cross_embedding_knn(self.emb, (first_labels, second_labels), knn_args = {'n_neighbors': 5})}")
            except:
                print("Can't calculate the Cross embedding")
        else:
            #Set all labels to be the same
            labels = np.ones(shape = (len(self.emb)))

        try:    
            print(f"FOSCTTM: {self.FOSCTTM(self.sim_diffusion_matrix[self.len_A:, :self.len_A])}") #This gets the off-diagonal part
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

    def plot_t_grid(self, rate = 3):
        """Returns the best T value according to its FOSCTTM score"""
        fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 16)) # Creates a grid of 2x5 for the subplots
        F_scores = np.array([])
        for i in range(1, 11):
            # Calculate the row and column index for the current subplot
            row = (i - 1) // 5
            col = (i - 1) % 5
            
            # Perform the diffusion operation
            a = i * rate
            diffused_array, projectionAB, projectionBA = self.get_diffusion(self.similarity_matrix, a)

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


