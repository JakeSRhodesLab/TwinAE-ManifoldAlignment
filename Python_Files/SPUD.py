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
  def __init__(self, knn = 5, operation = "normalize", verbose = 0, **kwargs):
        '''
        Creates a class object. 
        
        Arguments:
          :Knn: states how many nearest neighbors we want to use in the graph construction. If
            Knn is set to "connect" then it will ensure connection in the graph.

          :Operation: States the method of how we want to adjust the off-diagonal blocks in the alignment. 
            It can be 'sqrt', 'log', any float, or 'None'.
            If 'sqrt', it applies a square root function, and then transposes it to start at 0.
            If 'log', it applies a natural log, and then gets the distances between each point. Requires 1 to 1 correspondence.
            If 'float', it multiplies the off-diagonal block by the float value. 
            If 'None', it applies no additional transformation besides normalizing the values between 0 and 1. 
            
          :verbose: can be any float or integer. Determines what is printed as output as the function runs.

          :**kwargs: key word values for the graphtools.Graph function. 
          '''

        #Set the values
        self.verbose = verbose
        self.knn = knn
        self.operation = operation
        self.kwargs = kwargs

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

        #Check to make sure that the domains are the same size if we are using abs or log
        if (self.operation == "abs" or self.operation == "log") and len(dataA) != len(dataB):
            raise AssertionError('The operation "abs" and log "only" works with a one-to-one correspondence.')

        #For each domain, calculate the distances within their own domain
        self.distsA = self.get_SGDM(dataA)
        self.distsB = self.get_SGDM(dataB)

        #Create Igraphs from the input.
        self.graphA = graphtools.Graph(self.distsA, knn = self.knn, knn_max= self.knn, **self.kwargs).to_igraph() 
        self.graphB = graphtools.Graph(self.distsB, knn = self.knn, knn_max= self.knn, **self.kwargs).to_igraph()

        #Cache these values for fast lookup
        self.len_A = self.graphA.vcount()
        self.len_B = self.graphB.vcount()

        #Save the known Anchors
        self.known_anchors = np.array(known_anchors)

        #Merge the graphs 
        self.graphAB = self.merge_graphs()

        #Get the distances
        self.block = self.get_block(self.graphAB)

  """<><><><><><><><><><><><><><><><><><><><>     HELPER FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
  def normalize_0_to_1(self, value):
    """Normalizes the value to be between 0 and 1 and resets infinite values."""

    #Scale it and check to ensure no devision by 0
    if np.max(value[~np.isinf(value)]) != 0:
      value = (value - value.min()) / (value[~np.isinf(value)].max() - value.min()) 

    #Reset inf values
    value[np.isinf(value)] = 1

    return value

  def get_SGDM(self, data):
    """SGDM - Same Graph Distance Matrix.
    This returns the normalized distances within each domain."""

    #Just using a normal distance matrix without Igraph
    dists = squareform(pdist(data))

    #Normalize it and return the data
    return self.normalize_0_to_1(dists)

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
      
  def FOSCTTM(self, Wxy): 
        """
        FOSCTTM stands for average Fraction of Samples Closer Than the True Match.
        
        Lower scores indicate better alignment, as similar or corresponding points are mapped closer 
        to each other through the alignment process. If a method perfectly aligns all corresponding 
        points, the average FOSCTTM score would be 0. 

        Wxy should be either off-diagonal portion (that represents mapping from one domain to the other)
        of the block matrix. 
        """
        n1, n2 = np.shape(Wxy)
        if n1 != n2:
            raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

        dists = Wxy

        nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
        nn.fit(dists)

        _, kneighbors = nn.kneighbors(dists)

        return np.mean([np.where(kneighbors[i, :] == i)[0] / n1 for i in range(n1)])
  
  """<><><><><><><><><><><><><><><><><><><><>     PRIMARY FUNCTIONS BELOW     <><><><><><><><><><><><><><><><><><><><>"""
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

    #Get the vertices to find the distances between
    verticesA = np.array(range(self.len_A))
    verticesB = np.array(range(self.len_B))

    #Get the off-diagonal block by using the distance method
    off_diagonal = self.normalize_0_to_1(np.array(graph.distances(source = verticesA, target = verticesB, weights = "weight", algorithm = "dijkstra")))

    #Apply modifications to operation differences
    if type(self.operation) == float:
      off_diagonal *= self.operation

    if self.operation == "sqrt":
      off_diagonal = np.sqrt(off_diagonal + 1)
      off_diagonal = off_diagonal - off_diagonal.min()

    if self.operation == "log":
      if len(verticesA) != len(verticesB):
        print("Cannot compute the log modification due to different domain sizes. Proceeding with no modification.")

      else:
        off_diagonal = self.normalize_0_to_1((squareform(pdist((-np.log(1+off_diagonal)))))) #NOTE: adding one seems to work the best

    #Create the block
    block = np.block([[self.distsA, off_diagonal],
                      [off_diagonal.T, self.distsB]])

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
        ax.set_title("SPUD")

        #To plot line connections
        if show_lines:
            for i in range(self.len_B):
                ax.plot([self.emb[0 + i, 0], self.emb[self.len_A + i, 0]], [self.emb[0 + i, 1], self.emb[self.len_A + i, 1]], color = 'lightgrey', alpha = .5)

        #Put black dots on the Anchors
        if show_anchors:
            ax.scatter(self.emb[self.known_anchors, 0], self.emb[self.known_anchors, 1], s = 10, color = 'black', marker="s")
        
        #Show plot
        plt.show()
