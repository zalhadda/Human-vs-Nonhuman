###############
### IMPORTS ###
###############

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import sys

######################
### EXTRACT CORPUS ###
######################

'''
Reads a given file and returns all the lines found
'''

def Read(my_file):

    lines = None

    with open(my_file) as f:
        lines = f.readlines()

    return lines

# End of Read()

##############
### TF/IDF ###
##############

'''
Retrieves the tf/idf score of all the words in the provided corpus
'''

def Tf_idf(corpus):

    vectorizer = TfidfVectorizer(stop_words = 'english',  analyzer = 'word', max_df = 0.95, min_df = 0.001, ngram_range = (1, 5))
    
    # Transform the data

    X = vectorizer.fit_transform(corpus)

    # Perform K_means

    K_Means(X)

    # Convert the idf scores into a dictionary

    idf = vectorizer.idf_

    idf_dict = dict(zip(vectorizer.get_feature_names(), idf))

    return idf_dict

# End of Tf_idf()

#############################
### CLUSTERING & PLOTTING ###
#############################

'''
Clusters the data using KMeans
'''

def K_Means(scores):

    classifier = KMeans(n_clusters = 2, random_state = 0)

    labels = classifier.fit_predict(scores)

    # Plot Data

    # Make 2D coordinates from the scores matrix

    scores_dense = scores.todense()

    pca = PCA(n_components = 2).fit(scores_dense)

    coordinates = pca.transform(scores_dense)

    label_colours = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", 
                "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]

    colours = [label_colours[i] for i in labels]

    plt.scatter(coordinates[:, 0], coordinates[:, 1], c = colours)

    plt.savefig('pca_scatter_matrix.pdf', bbox_inches = 'tight')

    # Plot the cluster centres

    centroids = classifier.cluster_centers_

    centroids_coordinates = pca.transform(centroids)

    plt.scatter(centroids_coordinates[:, 0], centroids_coordinates[:, 1], marker = 'X', s = 200, linewidths = 2, c = '#444d60')

    #plt.show()

    plt.savefig('cluster_centres.pdf', bbox_inches = 'tight')


# End of K_Means

############
### MAIN ###
############

def Main():

    if (len(sys.argv) != 2):
        print("Incorrect # of parameters.")
        sys.exit()

    # It's expected that the second argument will have the text document's name

    my_file = str(sys.argv[1])

    # Creates the corpus
    # The corpus is all the lines extracted from the given file

    corpus = Read(my_file)

    # Retrieves tf-idf score

    score = Tf_idf(corpus)

    for key, value in score.items():
        print (str(key) + ": " + str(value))

# End of Main()

Main()
