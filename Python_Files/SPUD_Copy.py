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

class SPUD_Copy:
  def __init__(self, dataA, dataB, known_anchors, knn = 5, decay = 40, operation = "normalize"):
        '''dataA and dataB should simply just be the data. We will convert it
        to Igraph. Same as nama or mali

        Known Anchors should be in an array shape (n, 2), where n is the number of
        corresponding points and index [n, 0] is the node in dataA that corresponds to [n, 1]
        which is the node found in dataB

        Knn states how many nearest neighbors we want to use in the graph. If
        Knn is set to "connect" then it will ensure connection in the graph.

        Operation is the way we want to calculate to the distances: can do normalize,
        average, abs (for the distance between the two nodes), maximize or minimize. 
        TODO: In the future, maybe instead of maximize or minimize we can allow the user
        to input a float value and use that for greater flexibility. 

        Show is a boolean value. Set to True if you want to see the distance
        matrix.'''

        #Set the values
        self.decay = decay

        #Check to make sure that the domains are the same size
        if operation == "abs" and len(dataA) != len(dataB):
            raise AssertionError('The operation abs only works with a one-to-one correspondence.')
        
        self.operation = operation

        #Create Igraphs from the input.
        self.graphA = graphtools.Graph(self.get_SGDM(dataA), knn = knn, decay = self.decay, knn_max= knn, precomputed='affinity').to_igraph()
        self.graphB = graphtools.Graph(self.get_SGDM(dataB), knn = knn, decay = self.decay, knn_max= knn,  precomputed='affinity').to_igraph()

        #Cache these values for fast lookup
        self.len_A = self.graphA.vcount()
        self.len_B = self.graphB.vcount()

        #Making our known Anchors
        self.known_anchors = np.array(known_anchors)

        #Merge the graphs 
        self.graphAB = self.merge_graphs()

        #Get the distances
        self.block = self.get_block(self.graphAB)

  """HELPER FUNCTIONS"""
  def normalize_0_to_1(self, value):
    """Normalizes the value to be between 0 and 1 and resets infinite values"""

    #Scale it and check to ensure no devision by 0
    if np.max(value[~np.isinf(value)]) != 0:
      value = (value - value.min()) / (value[~np.isinf(value)].max() - value.min())

    #Reset inf values
    value[np.isinf(value)] = 1

    return value

  def get_SGDM(self, data):
    """SGDM - Same Graph Distance Matrix.
    This returns the normalized distances within each domain"""
    #Just using a normal distance matrix without Igraph
    dists = squareform(pdist(data))

    #Normalize it and return the data
    return self.normalize_0_to_1(dists)

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
  def merge_graphs(self): #NOTE: This process takes a significantly longer with more KNN (O(N) complexity)
        """Creates a new graph from A and B using the known_anchors
        and by merging them together"""

        #Change known_anchors to correspond to off diagonal matricies
        known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

        #Merge the two graphs together
        merged = self.graphA.disjoint_union(self.graphB) #Note: The distances between graphs may not be the same. It this is the case, would we want to scale the data first?

        #Now add the edges between anchors
        merged.add_edges(list(zip(known_anchors_adjusted[:, 0], known_anchors_adjusted[:, 1])))
        merged.es[-len(known_anchors_adjusted):]["weight"] = np.repeat(1, len(known_anchors_adjusted))

        #Return the Igraph object
        return merged
    
  def get_block(self, graph):
    """Returns a transformed and normalized block"""

    #Get the block
    block = np.array(graph.distances(weights = "weight", algorithm = "dijkstra"))

    #Cache off-diagonal block for efficiency and do methods based on operation:
    if self.operation == "normalize":
      #Normalize 0 to 1
      off_diagonal = self.normalize_0_to_1(block[:self.len_A, self.len_A:])

    elif self.operation == "average":
       off_diagonal = block[:self.len_A, self.len_A:] * 0.5

    elif self.operation == "minimize":
       #This should be chosen when we know there is a strong relationship between the two domains
       off_diagonal = block[:self.len_A, self.len_A:] * 0.2

    elif self.operation == "maximize":
       #This should be chosen when we know there is a weak relationship between the two domains
       off_diagonal = block[:self.len_A, self.len_A:] * 0.8

    elif self.operation == "abs":
      #Finds the off-diagonal by finding the how each domain adds to the off-diagonal then subtracting the two together
      block1 = (block[:self.len_A, self.len_A:] - block[:self.len_A, :self.len_A])
      block2 = (block[:self.len_A, self.len_A:] - block[self.len_A:, self.len_A:])
      off_diagonal = np.abs(block1 - block2)

    #Re-apply the transfomration to the block NOTE: the main diagonals are already normalized
    block[:self.len_A, self.len_A:] = off_diagonal
    block[self.len_A:, :self.len_A] = off_diagonal.T

    return block

  """VISUALIZATION FUNCTIONS"""
  def plot_graphs(self):
    #Plot the graph connections
    fig, axes = plt.subplots(1, 3, figsize = (19, 12))
    ig.plot(self.graphA, vertex_color=['green'], target=axes[0], vertex_label= list(range(self.len_A)))
    ig.plot(self.graphB, vertex_color=['cyan'], target=axes[1], vertex_label= list(range(self.len_B)))
    ig.plot(self.graphAB, vertex_color=['orange'], target=axes[2], vertex_label= list(range(self.len_A + self.len_B)))
    axes[0].set_title("Graph A")
    axes[1].set_title("Graph B")
    axes[2].set_title("Graph AB")
    plt.show()
  
  def plot_heat_map(self):
    #Plot the block matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(self.block, cmap='viridis', mask = (self.block > 50))
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
