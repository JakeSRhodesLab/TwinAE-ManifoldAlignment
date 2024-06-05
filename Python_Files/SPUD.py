#Shortest Path to Union Domains (SPUD)

#Install the libraries
from scipy.spatial.distance import pdist, squareform
import graphtools
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import seaborn as sns

class SPUD:
  def __init__(self, dataA, dataB, known_anchors, knn = 5, decay = 40, operation = "average", show = False, kind = "pure"):
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
        matrix.

        Kind is telling what kind of distance matrix you want.'''

        #Set the values
        self.decay = decay
        self.operation = operation
        self.show = show
        self.kind = kind


        #Create Igraphs from the input.
        if knn == "connect": #TODO: implement the similarity function
          self.graphA = self.construct_connected_graph(dataA)
          self.graphB = self.construct_connected_graph(dataB)

        else:
          Gx = graphtools.Graph(dataA, knn = knn, decay = self.decay, knn_max= knn) 
          Gy = graphtools.Graph(dataB, knn = knn, decay = self.decay, knn_max= knn)

          #Create the igraph distances
          self.graphA = Gx.to_igraph()
          self.graphB = Gy.to_igraph()

        #Create Same Graph Distance Matricies by direct connections
        self.pure_matrix_A, self.pure_matrix_B = self.get_pure_distance(dataA, dataB)
        self.pure_similar_A, self.pure_similar_B = self.get_pure_similarities(dataA, dataB)

        #Making our known Anchors
        self.known_anchors = np.array(known_anchors)
        self.known_anchors_A = self.known_anchors.T[0]
        self.known_anchors_B = self.known_anchors.T[1]

        #Create node paths and edge paths
        self.node_paths_A = self.make_node_paths(self.graphA, self.known_anchors_A)
        self.node_paths_B = self.make_node_paths(self.graphB, self.known_anchors_B)

        self.edge_paths_A = self.make_edge_paths(self.graphA, self.known_anchors_A)
        self.edge_paths_B = self.make_edge_paths(self.graphB, self.known_anchors_B)

        #Get Same Graph Distance matricies by following the closest neighbor paths
        self.matrix_A, self.scales_A = self.get_SGDM_and_scale(self.graphA)
        self.matrix_B, self.scales_B = self.get_SGDM_and_scale(self.graphB)

        self.matrix_AB = self.get_DGDM()

        #Finally, get the block matrix
        self.block = self.get_block_matrix()

  """HELPER FUNCTIONS"""
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

  def get_pure_distance(self, x, y):
    """This returns the normalized distances within each domain"""
    #Just using a normal distance matrix without Igraph
    x_dists = squareform(pdist(x))
    y_dists = squareform(pdist(y))

    #normalize it
    x_dists = x_dists / np.max(x_dists, axis = None)
    y_dists = y_dists / np.max(y_dists, axis = None)

    return x_dists, y_dists
  
  def get_pure_similarities(self, x, y):
    """This function creates graphs based on the similarity. These should be
    between all known points"""

    #Create graphs using the highest KNN possible
    similarity_graph_A = graphtools.Graph(x, knn = len(x)-2, decay = 1) #We do minus two because that is the max it allows
    similarity_graph_B = graphtools.Graph(y, knn = len(y)-2, decay = 1)

    #Return the similarity graph... note they aren't Igraphs... :)
    return (1- similarity_graph_A.K.toarray()), (1- similarity_graph_B.K.toarray())

  def make_node_paths(self, graph, anchors):
    """Will return all the node paths in a list"""

    #Create a blank list to append to the master one
    node_paths = []

    #For each node in graph get the shortest path
    for i in range(graph.vcount()):
      node_paths.append(graph.get_shortest_paths(i, to=anchors, algorithm = "dijkstra", weights=None, output="vpath"))

    return node_paths

  def make_edge_paths(self, graph, anchors):
    """Will return all the edge paths in a list"""

    #Create a blank list to append to the master one
    edge_paths = []

    #For each node in graph get the shortest path
    for i in range(graph.vcount()):
      edge_paths.append(graph.get_shortest_paths(i, to=anchors, weights=None, output="epath")) #It works without weights

    return edge_paths

  def get_SGDM_and_scale(self, graph):
    """Get Same Graph Distance Matrix. It will return a distance matrix for all
    nodes to each other node within the same graph.

    It also returns the scale (which is just the maximum value)."""

    #Get the entire distance matrix
    distance_matrix = np.array(graph.distances(weights = "weight")) #It seems when weights are set to none it made no difference.... :)
    inf_bool_matrix = np.isinf(distance_matrix)

    #Flatten both lists
    distance_matrix_flat = distance_matrix.flatten()
    inf_bool_matrix_flat = inf_bool_matrix.flatten()

    #Get the maximum
    scale = np.max(distance_matrix_flat[~inf_bool_matrix_flat])

    #Normalize it
    distance_matrix = (distance_matrix/scale) #- 0.5 #I read it that it could run best if the mean = 0.

    #Set inf values to 2... This will make it so when the DGDM runs it will use those two values -> this seems to help our models predictions
    distance_matrix[np.isinf(distance_matrix)] = 1 # Set to one because it 1 is the maximum distance... Maybe we should test 2 out

    return distance_matrix, scale
  
  """THE PRIMARY FUNCTIONS"""
  def get_real_DM(self, nodePaths, pureDistanceMatrix): #NOTE: This is currently finding the path to its nearest anchor, and not necessarily the right anchor to connect with the other graph
    """Get Same Graph Distance Matrix by going through each node path and adding each distance.
    It will only return the distance to the closest anchor"""
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

    #Scale it and check to ensure no devision by 0
    if np.max(node_distance_matrix[~np.isinf(node_distance_matrix)]) != 0:
      node_distance_matrix = node_distance_matrix / np.max(node_distance_matrix[~np.isinf(node_distance_matrix)])

    #Reset inf values
    node_distance_matrix[np.isinf(node_distance_matrix)] = 1

    return node_distance_matrix

  def AvgTwoNodes(self, k, l):
    '''K should be the node we want in graphA, and L should be the node in
    graphB. It will find the shortest distance between the two.

    This will use already scaled values.
    '''

    #Get the distance from K to all anchors
    k_to_all_anchor_dist = self.matrix_A[k][self.known_anchors_A] #Note, this is already scaled

    #Get the distance from L to all anchors
    l_to_all_anchor_dist = self.matrix_B[l][self.known_anchors_B] #Note, this is already scaled


    #Average metric. Seems to do well, but points similar to each other end up far away but on similar places
    if self.operation == "average":
      average_array = (k_to_all_anchor_dist + l_to_all_anchor_dist) / 2

      #else we want the shortest distance
      return np.min(average_array)

    #Abs value metric -- I think this is the best
    elif self.operation == "abs":
      #THIS WAY CHOOSES THE CLOSEST ANCHOR TO ONE POINT, AND THEN USES THAT ANCHOR (It compares both of them)

      #Beause we don't want both nodes to go to a far away anchor which might be a similar distance, we force it to closest ancor
      k_smallest_index = np.argmin(k_to_all_anchor_dist)
      abs_array = np.abs(k_to_all_anchor_dist[k_smallest_index] - l_to_all_anchor_dist[k_smallest_index])

      #This includes going the other direction too
      l_smallest_index = np.argmin(l_to_all_anchor_dist)

      abs_array = np.append(abs_array, np.abs(k_to_all_anchor_dist[l_smallest_index] - l_to_all_anchor_dist[l_smallest_index])) #Note: probably a faster appraoch than this

      return np.min(abs_array)

  def get_DGDM(self):
    """DGDM stands for Different graphs distance Matrix
    This gets the distance matrix between two graphs (or the same graph). Graph 1 should be the first
    graph and graph 2 should be the second that we want to compare distances too

    For each node in graph 1, it will find the distance to each node in graph 2"""

    """For each point, find the average distance from node x to node y"""
    #Create an array of 0s we can fill
    XY = np.zeros(shape=(self.graphA.vcount(), self.graphB.vcount()))

    #Depending on how we want to calculate
    if self.kind == "distance":
      #Create the nodes to anchor distances for each graph
      self.all_dist_to_anchors_A = self.get_real_DM(self.node_paths_A, self.pure_matrix_A)
      self.all_dist_to_anchors_B = self.get_real_DM(self.node_paths_B, self.pure_matrix_B)


      #Add each value together. A[:, np.newaxis, :] + B[np.newaxis, :, :]
      if self.operation == "average":
        all_anchors = (self.all_dist_to_anchors_A[:, np.newaxis, :] + self.all_dist_to_anchors_B[np.newaxis, :, :]) #/2 NOTE: I removed the /2 and it looks better -- Not too tested though
        return all_anchors.min(axis=2) #use the smallest distance

      elif self.operation == "abs": #The all method works... but the new approach here isn't as good
        #return (np.abs(self.all_dist_to_anchors_A[:, np.newaxis, :] - self.all_dist_to_anchors_B[np.newaxis, :, :])).min(axis=2)

        #Because we don't want to go through two far away nodes... so just do the closest one... if not we can just do this line: return np.abs(self.all_dist_to_anchors_A[:, np.newaxis, :] - self.all_dist_to_anchors_B[np.newaxis, :, :])
        A_smallest_index = self.all_dist_to_anchors_A.argmin(axis=1)
        B_smallest_index = self.all_dist_to_anchors_B.argmin(axis=1)

        #We want it to be in shape 150 by 150
        A_smallest_index = np.tile(A_smallest_index, (self.graphA.vcount(), 1))
        B_smallest_index = np.tile(B_smallest_index, (self.graphB.vcount(), 1)).T

        #Now do the actuall math and absolute value part. If we want to use all anchors, we can simply return this line.
        all_anchors = np.abs(self.all_dist_to_anchors_A[:, np.newaxis, :] - self.all_dist_to_anchors_B[np.newaxis, :, :])

        # Generate row and column index grids
        row_indices, col_indices = np.meshgrid(np.arange(self.graphA.vcount()), np.arange(self.graphB.vcount()), indexing='ij') #This might error if b != a

        #This line creates a third matrix with shape (150, 150) with the smallest of the two arrays
        return np.minimum(all_anchors[row_indices, col_indices, A_smallest_index], all_anchors[row_indices, col_indices, B_smallest_index])
    else: #This runs for both pure distance and similarity
      #Loop over each x
      for x_index in range(self.graphA.vcount()):
        #loop over each y
        for y_index in range(self.graphB.vcount()):
          #Do the math
          XY[x_index][y_index] = self.AvgTwoNodes(x_index, y_index)

      #Returns the matrix
      return XY

  #Note: We could add logic to block_matrix to choose between pure_matrix_A and matrix_A when the graph is best. Unsure how this would translate to other datasets
  def get_block_matrix(self):
    """Returns the block matrix."""

    if self.kind == "pure": #Great when KNN is low or around 5
      block_matrix = np.block([[self.pure_matrix_A, self.matrix_AB],
                               [self.matrix_AB.T, self.pure_matrix_B]]) #Note: if we just use matrix_A instead of pure, the results can vary for better or worse. This seems to be better most of the time

    elif self.kind == "similarity": #Litterally amazing :)
      block_matrix = np.block([[self.pure_similar_A, self.matrix_AB],
                               [self.matrix_AB.T, self.pure_similar_B]])

    elif self.kind == "distance":
      block_matrix = np.block([[self.pure_matrix_A, self.matrix_AB], #interestingly, it does better with the pure
                               [self.matrix_AB.T, self.pure_matrix_B]])

    else:
      print(f"Did not understand your input for kind = {self.kind}. /n Please use 'pure', 'distance' or 'similarity'")
      return None

    #Plot the block matrix
    if self.show:
      plt.figure(figsize=(10, 8))
      sns.heatmap(block_matrix, cmap='viridis', mask = (block_matrix > 4))
      plt.title('Block Matrix')
      plt.xlabel('Graph A Vertex')
      plt.ylabel('Graph B Vertex')
      plt.show()

    return block_matrix

  """VISUALIZATION FUNCTIONS"""
  def veiw_graphs(self):
    fig, axes = plt.subplots(1, 2, figsize = (13, 10))
    ig.plot(self.graphA, vertex_color=['green'], target=axes[0], vertex_label= list(range(self.graphA.vcount())))
    ig.plot(self.graphB, vertex_color=['cyan'], target=axes[1], vertex_label= list(range(self.graphB.vcount())))
    axes[0].set_title("Graph A")
    axes[1].set_title("Graph B")

  def graph_node_path(self, nodeP):
    '''NodeP should be a list equal to size of graphs. Each value represents
    which node you want to visualize.

    For example. If we want to graph both graphA and graphB, then
    NodeP = [[2,4], 3], it would plot the paths from node 2 to anchor value 4
    in gx, and node 3 to all anchor points in gy.
    '''


    #Loop over each graph
    for graph in [self.graphA, self.graphB]:
      #Create Flat node paths
      if graph == self.graphA:
        try:
          #This code will work if we are given a nested list. Else it will error
          flat_node_path = [item for sublist in self.node_paths_A[nodeP[0]] for item in sublist]
          flat_edge_path = [item for sublist in self.edge_paths_A[nodeP[0]] for item in sublist]

        except:
          #This code will run if we are given a single list
          flat_node_path = self.node_paths_A[nodeP[0][0]][nodeP[0][1]]
          flat_edge_path = self.edge_paths_A[nodeP[0][0]][nodeP[0][1]]

      else:
        try:
          flat_node_path = [item for sublist in self.node_paths_B[nodeP[1]] for item in sublist]
          flat_edge_path = [item for sublist in self.edge_paths_B[nodeP[1]] for item in sublist]

        except:
          flat_node_path = self.node_paths_B[nodeP[1][0]][nodeP[1][1]]
          flat_edge_path = self.edge_paths_B[nodeP[1][0]][nodeP[1][1]]

      #This hides what we dont want to see
      graph.vs["size"] = 0
      graph.vs["color"] = "white"
      graph.vs["label"] = None
      graph.es["color"] = "white"
      graph.es["width"] = 0

      #Change the label size
      graph.vs["label_size"] = 10

      #Chaning the values of nodes we do want to see
      colorChange = 0

      #Loops through each vertex to change it
      for v in flat_node_path:
          if v == flat_node_path[0]:
            colorChange = 0

          graph.vs[v]["size"] = 38

          if graph == self.graphA:
            graph.vs[v]["color"] = "rgb(0, " + str(colorChange) + ", 155)" #This has the original as pink, and the farther away the more green it gets
            graph.vs[v]["label"] = "A:" +str(v)
          else:
            graph.vs[v]["color"] = "rgb(200, " + str(colorChange) + ", 155)"
            graph.vs[v]["label"] = "B:" +str(v)


          if colorChange > 160:
            graph.vs[v]["label_color"] = "black"
          else:
            graph.vs[v]["label_color"] = "white"

          colorChange += 25

      #Now loops through each edge to change it's visual effect
      for e in flat_edge_path:
          graph.es[e]["color"] = "red"
          graph.es[e]["width"] = 2

    ###Now show the graphs###

    #Make it pretty
    plt.figure(figsize=(12, 12))
    plt.title('Connecting two Node paths together')

    #Putting them on the same graph
    layout = self.graphA.layout("kamada_kawai")

    ig.plot(self.graphA, autocurve=True, layout=layout, target=plt.gca())
    ig.plot(self.graphB, autocurve=True, layout=layout, target=plt.gca())

    #Finally veiw the graph
    plt.show()
    #plt.close()

