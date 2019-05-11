###############
### IMPORTS ###
###############

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import numpy as np
import pandas as np

import sys

####################
### REMODEL DATA ###
####################

'''
Reads the contents of a line and converts it to a dictionary.
The file is expected to have a format of:
    key: value
'''

def Make_Dict(my_file):

    # Reads the lines

    lines = None

    with open(my_file) as f:
        lines = f.readlines()


    # Splits each line into tokens and adds them to the dictionary

    dictionary = {}

    for line in lines:
        
        words = line.split(": ")

        dictionary[words[0]] = words[1]

    return dictionary

# End of Make_Dict()

#############################
### CLUSTERING & PLOTTING ###
#############################

'''
Clusters the data using a KMeans algorithm.
Since we're only looking at 2 categories: humans vs nonhumans, 
the # of clusters = 2
'''

def K_Means(scores):

    k_means = KMeans(n_clusters=2, random_state=0).fit_predict(scores)

    print(k_means.labels_)

# End of KMeans()

############
### MAIN ###
############

def Main():

    if (len(sys.argv) != 4):
        print("Invalid # of parameters.")
        print("Usage: plot_top_terms.py <human_scores_file> <nonhuman_scores_file> <both_scores_file>\n.")
        sys.exit()

    human_scores_file = sys.argv[1]
    nonhuman_scores_file = sys.argv[2]
    aggregate_scores_file = sys.argv[3]

    # Turn the scores in the files to dictionaries

    human_scores_dict = Make_Dict(human_scores_file)
    nonhuman_scores_dict = Make_Dict(nonhuman_scores_file)
    aggregate_scores_dict = Make_Dict(aggregate_scores_file)

    # Perform KMeans on the aggregate scores

    K_Means(aggregate_scores_dict)

# End of Main()

Main()
