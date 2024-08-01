#Shortest Path to Union Domains (SPUD)

#Install the libraries
from scipy.spatial.distance import pdist, squareform, _METRICS
import graphtools
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from sklearn.manifold import MDS
import seaborn as sns
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from rfgap import RFGAP

#Not necessary libraries, but helpful
from time import time

class SPUD:
  def __init__(self, distance_measure_A = "euclidean", distance_measure_B = "euclidean", knn = 5, operation = "normalize", IDC = 1, algorithm = "djikstra", verbose = 0, **kwargs):
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

          :Operation: States the method of how we want to adjust the off-diagonal blocks in the alignment. 
            It can be 'sqrt', 'log', any float, or 'None'.
            If 'sqrt', it applies a square root function, and then transposes it to start at 0. Best for when domains aren't the same shape.
            If 'log', it applies a natural log, and then gets the distances between each point. Requires 1 to 1 correspondence.
            If 'float', it multiplies the off-diagonal block by the float value. 
            If 'None', it applies no additional transformation besides normalizing the values between 0 and 1. 

          :IDC: stands for Inter-domain correspondence. It is the similarity value for anchors points between domains. Often, it makes sense
            to set it to be maximal (IDC = 1) although in cases where the assumptions (1: the corresponding points serve as alternative 
            representations of themselves in the co-domain, and 2: nearby points in one domain should remain close in the other domain) are 
            deemed too strong, the user may choose to assign the IDC < 1.
            
          :verbose: can be any float or integer. Determines what is printed as output as the function runs.

          :**kwargs: key word values for the graphtools.Graph function. 
          '''

        #Set the values
        self.distance_measure_A = distance_measure_A
        self.distance_measure_B = distance_measure_B
        self.verbose = verbose
        self.knn = knn
        self.operation = operation
        self.kwargs = kwargs
        self.IDC = IDC
        self.algorithm = algorithm

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

        #For each domain, calculate the distances within their own domain
        self.print_time()
        self.distsA = self.get_SGDM(dataA, self.distance_measure_A)
        self.distsB = self.get_SGDM(dataB, self.distance_measure_B)
        self.print_time(" Time it took to compute SGDM:  ")

        #Create Igraphs from the input.
        self.print_time()
        self.graphA = graphtools.Graph(self.distsA, knn = self.knn, knn_max= self.knn, **self.kwargs).to_igraph()
        self.graphB = graphtools.Graph(self.distsB, knn = self.knn, knn_max= self.knn, **self.kwargs).to_igraph()
        self.print_time(" Time it took to execute graphtools.Graph functions:  ")

        #Cache these values for fast lookup
        self.len_A = self.graphA.vcount()
        self.len_B = self.graphB.vcount()

        #Save the known Anchors
        self.known_anchors = np.array(known_anchors)

        #Merge the graphs 
        self.print_time()
        self.graphAB = self.merge_graphs()
        self.print_time(" Time it took to execute merge_graphs function:  ")

        #Get the distances
        self.print_time()
        self.block = self.get_block(self.graphAB)
        self.print_time(" Time it took to execute get_block function:  ")

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
    """SGDM - Same Graph Distance Matrix.
    This returns the normalized distances within each domain."""

    #Check to see if it is a function
    if callable(distance_measure):
      return distance_measure(data)

    #If the distances are precomputed, return the data. 
    elif distance_measure.lower() == "precomputed":
      return data
    
    #Euclidian
    elif distance_measure.lower() in _METRICS:
      #Just using a normal distance matrix without Igraph
      dists = squareform(pdist(data, metric = distance_measure.lower())) #Add it here -> if its in already for additionally block

    else:
      raise RuntimeError("Did not understand {distance_measure}. Please provide a function, or use strings 'precomputed', or provided by sk-learn.")

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
    
  def get_block(self, graph):
    """
    Returns a transformed and normalized block.
    
    Parameters:
      :graph: should be a graph that has merged together domains A and B.
      
    """

    #Get the vertices to find the distances between graphs. This helps when len_A != len_B
    verticesA = np.array(range(self.len_A))
    verticesB = np.array(range(self.len_B)) + self.len_A

    #Get the off-diagonal block by using the distance method. This returns a distnace matrix.
    off_diagonal = self.normalize_0_to_1(np.array(graph.distances(source = verticesA, target = verticesB, weights = "weight", algorithm = self.algorithm)))

    #Apply operation modifications
    if type(self.operation) == float:
      off_diagonal *= self.operation

    elif self.operation == "sqrt":
      off_diagonal = np.sqrt(off_diagonal + 1) #We have found that adding one helps

      #And so the distances are correct, we lower it so the scale is closer to 0 to 1
      off_diagonal = off_diagonal - off_diagonal.min()

    #If it is log, we check to to see if the domains match. If they do, we just apply the algorithm to the off-diagonal, which yeilds better results
    elif self.operation == "log" and self.len_A == self.len_B:
        #Apply the negative log, pdist, and squareform
        off_diagonal = self.normalize_0_to_1((squareform(pdist((-np.log(1+off_diagonal))))))

    #Create the block
    block = np.block([[self.distsA, off_diagonal],
                      [off_diagonal.T, self.distsB]])
    
    #If the operation is log, and the domain shapes don't match, we have to apply the process to the block. 
    if self.operation == "log" and self.len_A != self.len_B:
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
