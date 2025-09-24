# Plot Embeddings helper Function
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import Categorical


#Function that plots embeddings colored by label with lines connecting anchors
def plot_embeddings(emb, labels, anchors):
    #We are assuming the embeddings are half and half split (1 to 1 correspondence)
    styles = ['Domain A' if i < len(labels)/2 else 'Domain B' for i in range(len(emb))]

    plt.figure(figsize=(14, 8))
    ax = sns.scatterplot(x = emb[:, 0], y = emb[:, 1], style = styles, hue = Categorical(labels), s=120, markers= {"Domain A": "^", "Domain B" : "o"})

    #adjust anchors to match the reflected indicies
    known_anchors_adjusted = [(i, int(j + len(labels)/2)) for i, j in anchors]

    #Show lines between anchors
    for i in known_anchors_adjusted:
        ax.plot([emb[i[0], 0], emb[i[1], 0]], [emb[i[0], 1], emb[i[1], 1]], color = 'grey')
            
    plt.show()
