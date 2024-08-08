#Shortest Path to Union Domains (SPUD)

"""
Ideas to increase Computation:
np.tri to get the indicies seems to take a long time. We likely use the same indicies, so maybe we could create a dictionary of size to indicies?

Tasks: Go through code and check where we want to make things triangular, and where to test their speeds in computing. 
2. Go through and change the data type to a smaller float
3. Do I need the check for symmetric? IT always will be, unless given a precomputed data... ? No it will always be
3. With the mean and abs value, would we want to change from using kernals to Jaccard similarities? Maybe we could cap it at 3 connections so its still fast -- or we could apply diffusion and then overlap that with the kernal??  Or compute the kernal via numpy?
"""

#Install the libraries
from scipy.spatial.distance import pdist, squareform, _METRICS
import graphtools
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from sklearn.manifold import MDS
import seaborn as sns
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

#Not necessary libraries, but helpful
from time import time

class SPUD:
  def __init__(self, distance_measure_A = "euclidean", distance_measure_B = "euclidean", knn = 5,
               OD_method = "default", use_kernals = False, agg_method = "normalize", IDC = 1, 
               similarity_measure = "default", verbose = 0, **kwargs):
        '''
        Creates a class object. 
        
        Arguments:
          :distance_measure_A: Either a function, "precomputed" or SciKit_learn metric strings for domain A. If it is a function, then it should
            be formated like my_func(data) and returns a distance measure between points.
            If set to "precomputed", no transformation will occur, and it will apply the data to the graph construction as given. The graph
            function uses Euclidian distance, but this may manually changed through kwargs assignment.

          :distance_measure_B: Either a function, "precomputed" or SciKit_learn metric strings for domain B. If it is a function, then it should
            be formated like my_func(data) and returns a distance measure between points.
            If set to "precomputed", no transformation will occur, and it will apply the data to the graph construction as given. The graph
            function uses Euclidian distance, but this may manually changed through kwargs assignment.

          :Knn: states how many nearest neighbors we want to use in the graph construction. If
            Knn is set to "connect" then it will ensure connection in the graph.

          :OD_method: stands for Off-diagonal method. Can be the strings "abs", "mean" or "default". "Abs" calculates the absolute distances between the
            shortest paths to the same anchor, where as default calculates the shortest paths by traveling through an anchor. "Mean" calculates the average distance
            by going through each anchor.

          :agg_method: States the method of how we want to adjust the off-diagonal blocks in the alignment. 
            It can be 'sqrt', 'log', any float, or 'None'.
            If 'sqrt', it applies a square root function, and then transposes it to start at 0. Best for when domains aren't the same shape.
            If 'log', it applies a natural log, and then gets the distances between each point. Requires 1 to 1 correspondence.
            If 'float', it multiplies the off-diagonal block by the float value. 
            If 'None', it applies no additional transformation besides normalizing the values between 0 and 1. 

          :IDC: stands for Inter-domain correspondence. It is the similarity value for anchors points between domains. Often, it makes sense
            to set it to be maximal (IDC = 1) although in cases where the assumptions (1: the corresponding points serve as alternative 
            representations of themselves in the co-domain, and 2: nearby points in one domain should remain close in the other domain) are 
            deemed too strong, the user may choose to assign the IDC < 1.

          :similarity_measure: Can be default or Jaccard. Default uses the alpha decaying kernal to determine distances between nodes. Jaccard applies the jaccard similarity
            to the resulting graph. 
            
          :verbose: can be any float or integer. Determines what is printed as output as the function runs.

          :**kwargs: key word values for the graphtools.Graph function. 
          '''

        #Set the values
        self.distance_measure_A = distance_measure_A
        self.distance_measure_B = distance_measure_B
        self.distance_measure_A = distance_measure_A
        self.distance_measure_B = distance_measure_B
        self.verbose = verbose
        self.knn = knn
        self.agg_method = agg_method
        self.kwargs = kwargs
        self.IDC = IDC
        self.OD_method = OD_method.lower()
        self.use_kernals = use_kernals

        #Set self.emb to be None
        self.emb = None

  def fit(self, dataA, dataB, known_anchors):
        '''
        Does the work to compute the manifold alignment using shortest path distances. 
        
        Parameters:
          :dataA: the data for domain A. 
          :dataB: the data for domain B. 
          :known_anchors: this represents the points in domain A that correlate to domain B. Should be in a list
            formated like (n, 2), where n is the number of points that correspond. For any nth position, the 0th 
            place represents the point in domain A and the 1st position represents the point in domain B. Thus
            [[1,1], [4,3], [7,6]] would be appropiate.
        '''

        #Print timing data
        if self.verbose > 3:
           print("Time Data Below")

        #Cache these values for fast lookup
        self.len_A = len(dataA)
        self.len_B = len(dataB)

        #For each domain, calculate the distances within their own domain
        self.print_time()
        self.distsA = self.get_SGDM(dataA, self.distance_measure_A)
        self.distsB = self.get_SGDM(dataB, self.distance_measure_B)
        self.print_time("Time it took to compute SGDM:  ")

        #If these parameters are true, we can skip this all:
        if self.OD_method != "default" and self.use_kernals == False:
           if self.verbose > 0:
              print("Skipping graph creating. Performing nearest anchor manifold alignment (NAMA) instead of SPUD.")

        else:
          #Create Igraphs and kernals from the input.
          self.print_time()
          self.graphA = graphtools.Graph(self.reconstruct_symmetric(self.distsA), knn = self.knn, knn_max= self.knn, **self.kwargs)
          self.graphB = graphtools.Graph(self.reconstruct_symmetric(self.distsB), knn = self.knn, knn_max= self.knn, **self.kwargs)

          self.kernalsA = self.get_triangular(self.graphA.K.toarray())
          self.kernalsB = self.get_triangular(self.graphB.K.toarray())

          self.graphA = self.graphA.to_igraph()
          self.graphB = self.graphB.to_igraph()
          self.print_time("Time it took to execute graphtools.Graph functions:  ")

        #Save the known Anchors
        self.known_anchors = np.array(known_anchors)

        #Merge the graphs
        if self.OD_method == "default":
          if self.verbose > 0 and self.len_A > 1000:
             print("  --> Warning: Computing off-diagonal blocks will be exspensive. Consider setting OD_method to 'mean' or 'abs' for faster computation time.")


          self.print_time()
          self.graphAB = self.merge_graphs()
          self.print_time("Time it took to execute merge_graphs function:  ")

        #Get the distances
        self.print_time()
        self.block = self.get_block()
        self.print_time("Time it took to execute get_block function:  ")

        if self.verbose > 0:
           print("<><><><><><><><><><><><><>  Processed Finished  <><><><><><><><><><><><><>")

  """<><><><><><><><><><><><><><><><><><><><>     HELPER FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
  def print_time(self, print_statement =  ""):
    """A function that times the algorithms and returns a string of how
    long the function was last called."""

    #Only do this if the verbose is higher than 4
    if self.verbose > 3:

      #Start time. 
      if not hasattr(self, 'start_time'):
        self.start_time = time()

      #Check to see if it equals None
      elif self.start_time == None:
        self.start_time = time()

      else:
        #We need to end the time
        end_time = time()

        #Create a string to return
        time_string = str(round(end_time - self.start_time, 2))

        #Reset the start time
        self.start_time = None

        print(print_statement + time_string)

  def normalize_0_to_1(self, value):
    """Normalizes the value to be between 0 and 1 and resets infinite values."""

    #Scale it and check to ensure no devision by 0
    if np.max(value[~np.isinf(value)]) != 0:
      value = (value - value.min()) / (value[~np.isinf(value)].max() - value.min()) 

    #Reset inf values
    value[np.isinf(value)] = 1

    return value

  def get_SGDM(self, data, distance_measure):
  def get_SGDM(self, data, distance_measure):
    """SGDM - Same Graph Distance Matrix.
    This returns the normalized distances within each domain."""

    #Check to see if it is a function
    if callable(distance_measure):
      return distance_measure(data)
    if callable(distance_measure):
      return distance_measure(data)

    #If the distances are precomputed, return the data. 
    elif distance_measure.lower() == "precomputed":
    elif distance_measure.lower() == "precomputed":
      return data
    
    #Euclidian
    elif distance_measure.lower() in _METRICS:
      #Just using a normal distance matrix without Igraph
      dists = squareform(pdist(data, metric = distance_measure.lower())) #Add it here -> if its in already for additionally block

    else:
      raise RuntimeError("Did not understand {distance_measure}. Please provide a function, or use strings 'precomputed', or provided by sk-learn.")

    #Normalize it and return the data
    return self.get_triangular(self.normalize_0_to_1(dists))

  def get_off_diagonal_distances(self):
    """
    Calculates the off-diagonal by finding the closest anchors to each other.
    """

    #The algorithm uses the kernals for speed and efficiency (so we don't waste time calculating similarities twice.)
    if self.verbose > 2:
       print(f"Preforming {self.OD_method} calculations.\n")
    
    if self.use_kernals == True:
       matrixA = 1 - self.kernalsA
       matrixB = 1 - self.kernalsB
    else:
       matrixA = self.distsA
       matrixB = self.distsB

    if self.OD_method == "abs":
      """
      Excessive Memory Problem Solution: Batches

      We normally hit the problem with the Anchor Dists (though it may be the closest anchor part too -- hopefully not)

      Instead of selecting the anchor distances later we can apply it before via a loop (Maybe do bacthes of len_B/15?

      Can we shrink the float size? 
      
      """

      #Subset A and B to only the columns so we only have the distances to the anchors
      anchor_dists_A, indiciesA = self.index_triangular(matrixA, columns = self.known_anchors[:, 0], return_indices=True)
      anchor_dists_B, indiciesB = self.index_triangular(matrixB, columns = self.known_anchors[:, 1], return_indices=True)

      # Find the indices of the closest anchors for each node in both graphs
      A_smallest_index = self.min_bincount(indiciesA, anchor_dists_A)
      B_smallest_index = self.min_bincount(indiciesB, anchor_dists_B)

      #Strecth A and B to be the correct sizes, and then select the subtraction anchors
      anchor_dists_A = np.repeat(matrixA[A_smallest_index], repeats=self.len_B)
      anchor_dists_B = np.tile(matrixB[A_smallest_index], self.len_A)

      off_diagonal_using_A_anchors = np.abs(anchor_dists_A - anchor_dists_B)

      #Strecth A and B to be the correct sizes, and then select the subtraction anchors
      anchor_dists_A = np.repeat(matrixA[B_smallest_index], repeats=self.len_B)
      anchor_dists_B = np.tile(matrixB[B_smallest_index], self.len_A)

      off_diagonal_using_B_anchors = np.abs(anchor_dists_A - anchor_dists_B)

      #Perform the calculation
      off_diagonal = np.reshape(np.minimum(off_diagonal_using_A_anchors, off_diagonal_using_B_anchors), newshape=(self.len_A, self.len_B))

    if self.OD_method == "mean":

      #Take the mean of each one first, then select.
      anchor_dists_A = self.get_triangular_mean(*self.index_triangular(matrixA, columns = self.known_anchors[:, 0], return_indices=True)) 
      anchor_dists_B = self.get_triangular_mean(*self.index_triangular(matrixB, columns = self.known_anchors[:, 1], return_indices=True))

      #Strecth A and B to be the correct sizes, and so each value matches up with each other value.
      anchor_dists_A = np.repeat(anchor_dists_A, repeats= self.len_B)
      anchor_dists_B = np.tile(anchor_dists_B, self.len_A)
      
      #Convert it to the square matrix. NOTE: Do we want to convert it back into a triangular????? - Probably not.
      off_diagonal = np.reshape(np.abs(anchor_dists_A - anchor_dists_B), newshape=(self.len_A, self.len_B))
             
    return off_diagonal

  def get_triangular(self, matrix, tol=1e-4):
    """If the matrix is symmetric, the function seeks to save memory by cutting out information that is
    redundant. 
    """
    #Check if the matrix is symetric
    if np.allclose(matrix, matrix.T, atol=tol):
      
      #flatten the array and just take the upper triangular part
      return matrix[np.triu_indices_from(matrix)]
    
    else:
      #Return the original matrix if the matrix is not symmetric
      if self.verbose > 1:
          print("  Matrix is not symmetric. Failed to create sparse matrix.")
      return matrix

  def index_triangular(self, upper_triangular, columns, return_indices=False):
    """Indexes the triangular matrix. If rows or columns are set to None, it returns all of them.
    If return_indices is True, it also returns the indices and the mask used for indexing."""

    # Check to see if the ndim = 1, else it's already built
    if upper_triangular.ndim == 1:

        # Get the size of the original matrix
        size = int((-1 + np.sqrt(1 + (8 * upper_triangular.size))) // 2)
        indices = np.triu_indices(size)

        # Create the mask for the specified columns
        col_mask = np.isin(indices[1], columns)

        # Create the mask for the symmetric counterparts in the lower triangle
        row_mask = np.isin(indices[0], columns)

        # Combine the row_mask with the off_diagonal_mask
        row_mask = row_mask & (indices[0] != indices[1])


        # Apply the combined mask to indices
        indices = np.concatenate((indices[0][col_mask], indices[1][row_mask]))
        upper_triangular = np.concatenate((upper_triangular[col_mask], upper_triangular[row_mask]))

        if return_indices:
            # Return the matrix, its indices, and mask
            return upper_triangular, indices
        else:
            # Just return the matrix
            return upper_triangular
        
    else:

        # If the input is already a matrix, apply the row and column selection directly
        upper_triangular = upper_triangular[:, columns]

        return upper_triangular
    
  def get_triangular_mean(self, upper_triangular, indices):
    """Calculates the mean of the upper-triangle in a highly efficient and vectorized fashion. 
      Column can be set to True to calculate the mean of the column or False for rows"""
    
    #Incase some indicies were skipped. TODO: upgrade this for when the labels aren't continuous
    indices -= indices.min()

    # Calculate the sums and counts for each index
    sums = np.bincount(indices, weights=upper_triangular)
    counts = np.bincount(indices)

    return sums / counts
  
  def get_triangular_min(self, upper_triangle, indices):
    """Upper_triangle is a subsetted and flattned upper_triangle of a matrix
       Indicies tell us which values are kept and which values were subsetted out"""
    
    #Incase some indicies were skipped. TODO: upgrade this for when the labels aren't continuous
    indices -= indices.min()

    # For each row, return the argmin with the flattened upper triangle. 

    #


    return 

  def min_bincount(self, indices, values):
    # Get the unique indices and their positions
    unique_indices= np.unique(indices)
    
    # Initialize an array to store the minimum values and their positions
    min_values = np.full(unique_indices.shape, np.inf)
    
    # Use np.minimum.at to update the min_values array
    np.minimum.at(min_values, indices, values)

    # Create a mask for the minimum values
    min_pos = np.where((values == min_values[indices]))[0]

    # Extract the relevant indices
    list_thing = indices[min_pos]

    # Use numpy's unique function to find duplicates
    _, unique_indices = np.unique(list_thing, return_index=True)

    # Remove duplicates by selecting only the unique indice
    return min_pos[unique_indices]

  def reconstruct_symmetric(self, upper_triangular):
    """Rebuilds the triangular to a symmetric graph"""

    #Check to see if the ndim = 1, else its already built
    if upper_triangular.ndim == 1:

      #Cache size for faster processing
      size = int((-1 + np.sqrt(1 + (8 * upper_triangular.size))) // 2)

      #Create a matrix filled with zeros
      matrix = np.zeros((size, size))

      #Get the indicies for the upper trianglar
      indices = np.triu_indices(size)

      #Reset the matrix
      matrix[indices] = upper_triangular

      # Mirror the upper part to the lower part
      matrix[(indices[1], indices[0])] = upper_triangular 

      return matrix
    
    else:
       return upper_triangular
    
  
  """<><><><><><><><><><><><><><><><><><><><>     EVALUATION FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
  def cross_embedding_knn(self, embedding, Labels, knn_args = {'n_neighbors': 4}):
      """
      Returns the classification score by training on one domain and predicting on the the other.
      This will test on both domains, and return the average score.
      
      Parameters:
        :embedding: the manifold alignment embedding. 
        :Labels: a concatenated list of labels for domain A and labels for domain B
        :knn_args: the key word arguments for the KNeighborsClassifier."""

      (labels1, labels2) = Labels

      n1 = len(labels1)

      #initialize the model
      knn = KNeighborsClassifier(**knn_args)

      #Fit and score predicting from domain A to domain B
      knn.fit(embedding[:n1, :], labels1)
      score1 =  knn.score(embedding[n1:, :], labels2)

      #Fit and score predicting from domain B to domain A, and then return the average value
      knn.fit(embedding[n1:, :], labels2)
      return np.mean([score1, knn.score(embedding[:n1, :], labels1)])
      
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
  
  """<><><><><><><><><><><><><><><><><><><><>     PRIMARY FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
  def merge_graphs(self):
        """
        Creates a new graph from graphs A and B creating edges between corresponding points
        using the known anchors.
        """

        #Change known_anchors to correspond to off diagonal matricies
        self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

        #Merge the two graphs together
        merged = self.graphA.disjoint_union(self.graphB)

        #Now add the edges between anchors and set their  weight to 1
        merged.add_edges(list(zip(self.known_anchors_adjusted[:, 0], self.known_anchors_adjusted[:, 1])))
        merged.es[-len(self.known_anchors_adjusted):]["weight"] = np.repeat(self.IDC, len(self.known_anchors_adjusted))

        #Return the Igraph object
        return merged
    
  def get_block(self):
    """
    Returns a transformed and normalized block.
    
    Parameters:
      :graph: should be a graph that has merged together domains A and B.
      
    """

    #Find the off_diagonal block depending on our method
    if self.OD_method != "default":
       off_diagonal = self.get_off_diagonal_distances()
    else:
      #Get the vertices to find the distances between graphs. This helps when len_A != len_B
      verticesA = np.array(range(self.len_A))
      verticesB = np.array(range(self.len_B)) + self.len_A

      #Get the off-diagonal block by using the distance method. This returns a distnace matrix.
      #TODO: Think about how we neeed to reconstruct this
      off_diagonal = self.normalize_0_to_1(np.array(self.graphAB.distances(source = verticesA, target = verticesB, weights = "weight"))) # We could break this apart as another function to calculate the abs value in another way. This would reduce time complexity, though likely not be as accurate. 

    #Apply agg_method modifications
    if type(self.agg_method) == float:
      off_diagonal *= self.agg_method

    elif self.agg_method == "abs":
       pass


    elif self.agg_method == "sqrt":
      off_diagonal = np.sqrt(off_diagonal + 1) #We have found that adding one helps

      #And so the distances are correct, we lower it so the scale is closer to 0 to 1
      off_diagonal = off_diagonal - off_diagonal.min()

    #If it is log, we check to to see if the domains match. If they do, we just apply the algorithm to the off-diagonal, which yeilds better results
    elif self.agg_method == "log" and self.len_A == self.len_B:
        #Apply the negative log, pdist, and squareform
        off_diagonal = self.normalize_0_to_1((squareform(pdist((-np.log(1+off_diagonal))))))

    #Recreate the block matrix --> This may be faster?
    off_diagonal = self.reconstruct_symmetric(off_diagonal)

    #Create the block
    if self.use_kernals:
      block = np.block([[self.reconstruct_symmetric(1 - self.kernalsA), off_diagonal],
                        [off_diagonal.T, self.reconstruct_symmetric(1 - self.kernalsB)]])
    else:
      block = np.block([[self.reconstruct_symmetric(self.distsA), off_diagonal],
                        [off_diagonal.T, self.reconstruct_symmetric(self.distsB)]])
    
    #If the agg_method is log, and the domain shapes don't match, we have to apply the process to the block. 
    if self.agg_method == "log" and self.len_A != self.len_B:
      if self.verbose > 0:
         print("Domain sizes dont macth. Will apply the 'log' aggregation method against the whole block rather just the off_diagonal.")

      #Apply the negative log, pdist, and squareform
      block = self.normalize_0_to_1((squareform(pdist((-np.log(1+block))))))

    return block

  """VISUALIZATION FUNCTIONS"""
  def plot_graphs(self):
    """
    Using the Igraph plot function to plot graphs A, B, and AB. 
    """
    #Create figure
    fig, axes = plt.subplots(1, 3, figsize = (19, 12))

    #Plot each of the plots
    ig.plot(self.graphA, vertex_color=['green'], target=axes[0], vertex_label= list(range(self.len_A)))
    ig.plot(self.graphB, vertex_color=['cyan'], target=axes[1], vertex_label= list(range(self.len_B)))
    ig.plot(self.graphAB, vertex_color=['orange'], target=axes[2], vertex_label= list(range(self.len_A + self.len_B)))

    #Create the titles
    axes[0].set_title("Graph A")
    axes[1].set_title("Graph B")
    axes[2].set_title("Graph AB")

    plt.show()
  
  def plot_heat_map(self):
    """
    Plots the heat map for the manifold alignment. 
    """
    #Create the figure
    plt.figure(figsize=(8, 6))
    
    #Plot the heat map
    sns.heatmap(self.block, cmap='viridis', mask = (self.block > 50))

    #Add title and labels
    plt.title('Block Matrix')
    plt.xlabel('Graph A Vertex')
    plt.ylabel('Graph B Vertex')

    plt.show()

  def plot_emb(self, labels = None, n_comp = 2, show_lines = True, show_anchors = True, show_pred = False, show_legend = True, **kwargs): 
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
          #Create the mds object and then the embedding
          mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components= n_comp)
          self.emb = mds.fit_transform(self.block) 

        #Check to make sure we have labels
        if type(labels)!= type(None):
            #Seperate the labels into their respective domains
            first_labels = labels[:self.len_A]
            second_labels = labels[self.len_A:]

            #Calculate Cross Embedding Score
            try: #Will fail if the domains shapes aren't equal
                print(f"Cross Embedding: {self.cross_embedding_knn(self.emb, (first_labels, second_labels), knn_args = {'n_neighbors': 5})}")
            except:
                print("Can't calculate the Cross embedding")
        else:
            #Set all labels to be the same
            labels = np.ones(shape = (len(self.emb)))

        #Calculate FOSCTTM Scores
        try:    
            print(f"FOSCTTM: {self.FOSCTTM(self.block[self.len_A:, :self.len_A])}") #This gets the off-diagonal part
        except: #This will run if the domains are different shapes
            print("Can't compute FOSCTTM with different domain shapes.")

        #Create styles to change the points from graph 1 to be triangles and circles from graph 2
        styles = ['Domain A' if i < self.len_A else 'Domain B' for i in range(len(self.emb[:]))]

        #Create the figure
        plt.figure(figsize=(14, 8))

        #Imporrt pandas for the categorical function
        from pandas import Categorical

        #If show_pred is chosen, we want to show labels in Domain B as muted
        if show_pred:
            ax = sns.scatterplot(x = self.emb[self.len_A:, 0], y = self.emb[self.len_A:, 1], color = "grey", s=120, marker= "o", **kwargs)
            ax = sns.scatterplot(x = self.emb[:self.len_A, 0], y = self.emb[:self.len_A, 1], hue = Categorical(first_labels), s=120, marker= "^", **kwargs)
        
        else:
            #Now plot the points with correct lables
          ax = sns.scatterplot(x = self.emb[:, 0], y = self.emb[:, 1], style = styles, hue = Categorical(labels), s=120, markers= {"Domain A": "^", "Domain B" : "o"}, **kwargs)

        #Set the title and plot Legend
        ax.set_title("SPUD", fontsize = 25)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
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
             
        #Put black dots on the Anchors
        if show_anchors:
            
            #For each anchor set, plot lines between them
            for i in self.known_anchors_adjusted:
              ax.plot([self.emb[i[0], 0], self.emb[i[1], 0]], [self.emb[i[0], 1], self.emb[i[1], 1]], color = 'grey')
            
            #Create a new style guide so every other point is a triangle or circle
            styles2 = ['Domain A' if i % 2 == 0 else 'Domain B' for i in range(len(self.known_anchors)*2)]

            #Plot the black triangles or circles on the correct points
            sns.scatterplot(x = np.array(self.emb[self.known_anchors_adjusted, 0]).flatten(), y = np.array(self.emb[self.known_anchors_adjusted, 1]).flatten(), style = styles2,  linewidth = 2, markers= {"Domain A": "x", "Domain B" : "+"}, s = 45, color = "black")

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
            ax.set_title("Predicted Labels")

            plt.show()