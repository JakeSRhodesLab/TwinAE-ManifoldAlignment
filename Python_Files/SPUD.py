#Shortest Path to Union Domains (SPUD)

#Install the libraries
from scipy.spatial.distance import pdist, squareform
import graphtools
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from sklearn.manifold import MDS
import seaborn as sns
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

class SPUD:
  def __init__(self, dataA, dataB, known_anchors, knn = 5, decay = 40, operation = "average", verbose = 0):
        '''dataA and dataB should simply just be the data. We will convert it
        to Igraph. Same as nama or mali

        Known Anchors should be in an array shape (n, 2), where n is the number of
        corresponding points and index [n, 0] is the node in dataA that corresponds to [n, 1]
        which is the node found in dataB

        Knn states how many nearest neighbors we want to use in the graph. If
        Knn is set to "connect" then it will ensure connection in the graph.

        Operation is the way we want to calculate to the distances: can do
        average or abs (for the distance between the two nodes)

        Show is a boolean value. Set to True if you want to see the distance
        matrix.'''

        #Set the values
        self.decay = decay
        self.operation = operation
        self.verbose = verbose

        #Create Igraphs from the input.
        if knn == "connect": #TODO: implement the similarity function
          self.graphA = self.construct_connected_graph(dataA)
          self.graphB = self.construct_connected_graph(dataB)

        else:
          Ga = graphtools.Graph(dataA, knn = knn, decay = self.decay, knn_max= knn) 
          Gb = graphtools.Graph(dataB, knn = knn, decay = self.decay, knn_max= knn)

          #Create the igraph distances
          self.graphA = Ga.to_igraph()
          self.graphB = Gb.to_igraph()

        #Cache these values for fast lookup
        self.len_A = self.graphA.vcount()
        self.len_B = self.graphB.vcount()

        #Create Same Graph Distance Matricies by direct connections
        self.matrix_A = self.get_SGDM(dataA)
        self.matrix_B = self.get_SGDM(dataB)

        #Making our known Anchors
        self.known_anchors = np.array(known_anchors)

        #Create node paths and edge paths
        self.node_paths_A = self.make_node_paths(self.graphA, self.known_anchors.T[0])
        self.node_paths_B = self.make_node_paths(self.graphB, self.known_anchors.T[1])

        #Get the off-diagonal blocks
        self.matrix_AB = self.get_DGDM()

        #Finally, get the block matrix
        self.block = np.block([[self.matrix_A, self.matrix_AB], [self.matrix_AB.T, self.matrix_B]])

  """HELPER FUNCTIONS"""
  def normalize_0_to_1(self, value):
    """Normalizes the value to be between 0 and 1"""

    #Scale it and check to ensure no devision by 0
    if np.max(value[~np.isinf(value)]) != 0:
      value = (value - value.min()) / (value[~np.isinf(value)].max() - value.min())

    #Reset inf values
    value[np.isinf(value)] = 1

    return value

  def construct_connected_graph(self, graphData, initial_knn=2):
    """This function is called forces the graph to be fully connected"""
    connected = False
    while not connected:
        # Construct a k-nearest neighbor graph
        G = graphtools.Graph(graphData, knn = initial_knn, decay = self.decay, knn_max = initial_knn)

        # Convert it to to an igraph graph
        g = G.to_igraph()

        # Check if the graph is connected
        connected = g.is_connected()

        if not connected:
            initial_knn += 1  # Increase knn for the next iteration
        else:
            print(f"Graph is connected with knn={initial_knn}.")

    return g

  def get_SGDM(self, data):
    """SGDM - Same Graph Distance Matrix.
    This returns the normalized distances within each domain"""
    #Just using a normal distance matrix without Igraph
    dists = squareform(pdist(data))

    #Normalize it and return the data
    return self.normalize_0_to_1(dists)

  def make_node_paths(self, graph, anchors):
    """Will return all the node paths in a list"""

    #Create a blank list to append to the master one
    node_paths = []

    #For each node in graph get the shortest path
    for i in range(graph.vcount()):
      node_paths.append(graph.get_shortest_paths(i, to=anchors, algorithm = "dijkstra", weights=None, output="vpath"))

    return node_paths
  
  """EVALUATION FUNCTIONS"""
  def cross_embedding_knn(self, embedding, Y, knn_args = {'n_neighbors': 4}, other_side = True):
      (y1, y2) = Y

      n1, n2 = len(y1), len(y2)

      knn = KNeighborsClassifier(**knn_args)

      if other_side:
          knn.fit(embedding[:n1, :], y1)

          return knn.score(embedding[n1:, :], y2)

      else:
          #Train on other domain, predict on other domain ---- TODO
          knn.fit(embedding[n1:, :], y2)

          return knn.score(embedding[:n1, :], y1)
      
  def FOSCTTM(self, Wxy): #Wxy should be just the parrallel matrix
      n1, n2 = np.shape(Wxy)
      if n1 != n2:
          raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

      dists = Wxy

      nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
      nn.fit(dists)

      _, kneighbors = nn.kneighbors(dists)

      return np.mean([np.where(kneighbors[i, :] == i)[0] / n1 for i in range(n1)])
  
  """THE PRIMARY FUNCTIONS""" 
  def get_shortest_paths(self, nodePaths, pureDistanceMatrix): #NOTE: This is currently finding the path to its nearest anchor, and not necessarily the right anchor to connect with the other graph
    """Get Same Graph Distance Matrix by going through each node path and adding each distance.
    It returns a matrix of every node and the distances to their anchors"""

    #Create an empty list to return
    node_distance_matrix = []

    #Loop through Node paths
    for node in nodePaths:
      node_distance_to_anchors = []
      #Loop through each path
      for path in node: #We could just select the one that has the least connections to save time and computing power -- it will be close to the actuall but not the same
        #Check to make sure the path isn't empty
        if len(path) > 0:
          distance_to_anchor = 0

          #Now loop through the connections
          for index in range(len(path)-1):
            distance_to_anchor += pureDistanceMatrix[path[index]][path[index+1]]

          #We want to use the shortest one
          node_distance_to_anchors.append(distance_to_anchor)
        else: #Path size is infintie
          node_distance_to_anchors.append(np.inf)

      node_distance_matrix.append(node_distance_to_anchors)

    #Convert to an array
    node_distance_matrix = np.array(node_distance_matrix) #Question: code might run faster if we start it as an array

    return self.normalize_0_to_1(node_distance_matrix)
  
  def get_DGDM(self):
    """DGDM - Different Graphs Distance Matrix
    Finds the value of the off-diagonal blocks by traversing from graph through the anchor point to the other graph"""

    #Create the nodes to anchor distances for each graph
    self.all_dist_to_anchors_A = self.get_shortest_paths(self.node_paths_A, self.matrix_A) #Shaped (length of matrix, length of known anchors)
    self.all_dist_to_anchors_B = self.get_shortest_paths(self.node_paths_B, self.matrix_B)

    #Add each value together
    if self.operation == "average":
      all_anchors = (self.all_dist_to_anchors_A[:, np.newaxis, :] + self.all_dist_to_anchors_B[np.newaxis, :, :]) #/2 - It seems to work better without deviding
      return all_anchors.min(axis=2) #use the smallest distance

    elif self.operation == "abs":
      #return (np.abs(self.all_dist_to_anchors_A[:, np.newaxis, :] - self.all_dist_to_anchors_B[np.newaxis, :, :])).min(axis=2)

      #Because we don't want to go through two far away nodes... so just do the closest one... if not we can just do this line: return np.abs(self.all_dist_to_anchors_A[:, np.newaxis, :] - self.all_dist_to_anchors_B[np.newaxis, :, :])
      A_smallest_index = self.all_dist_to_anchors_A.argmin(axis=1)
      B_smallest_index = self.all_dist_to_anchors_B.argmin(axis=1)

      #We want it to be in shape 150 by 150
      A_smallest_index = np.tile(A_smallest_index, (self.len_A, 1))
      B_smallest_index = np.tile(B_smallest_index, (self.len_B, 1)).T

      #Now do the actuall math and absolute value part. If we want to use all anchors, we can simply return this line.
      all_anchors = np.abs(self.all_dist_to_anchors_A[:, np.newaxis, :] - self.all_dist_to_anchors_B[np.newaxis, :, :])

      # Generate row and column index grids
      row_indices, col_indices = np.meshgrid(np.arange(self.len_A), np.arange(self.len_B), indexing='ij') #This might error if b != a

      #This line creates a third matrix with shape (150, 150) with the smallest of the two arrays
      return np.minimum(all_anchors[row_indices, col_indices, A_smallest_index], all_anchors[row_indices, col_indices, B_smallest_index])
    
    else:
       raise RuntimeError('Operation not understood. Please use "average" or "abs"')

  """VISUALIZATION FUNCTIONS"""
  def plot_graphs(self):
    #Plot the graph connections
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    ig.plot(self.graphA, vertex_color=['green'], target=axes[0], vertex_label= list(range(self.len_A)))
    ig.plot(self.graphB, vertex_color=['cyan'], target=axes[1], vertex_label= list(range(self.len_B)))
    axes[0].set_title("Graph A")
    axes[1].set_title("Graph B")
    plt.show()
  
  def plot_heat_map(self):
    #Plot the block matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(self.block, cmap='viridis', mask = (self.block > 4))
    plt.title('Block Matrix')
    plt.xlabel('Graph A Vertex')
    plt.ylabel('Graph B Vertex')
    plt.show()

  def plot_emb(self, labels = None, n_comp = 2, show_lines = True, show_anchors = True, **kwargs): 
        """Creates and plots the embedding for ease"""

        #Convert to a MDS
        mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components= n_comp)
        self.emb = mds.fit_transform(self.block) #Later we should implent this to just be the block

        #Stress is a value of how well the emd did. Lower the better.
        print(f"Model Stress: {mds.stress_}")

        if type(labels)!= type(None):
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
            print(f"FOSCTTM: {self.FOSCTTM(self.block[self.len_A:, :self.len_A])}") #This gets the off-diagonal part
        except: #This will run if the domains are different shapes
            print("Can't compute FOSCTTM with different domain shapes.")

        #Veiw the manifold. Those shown as Triangles are from GX
        styles = ['graph 1' if i < self.len_A else 'graph 2' for i in range(len(self.emb[:]))]
        plt.figure(figsize=(14, 8))

        #Now plot the points
        import pandas as pd
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
