The Random Forest Classifier is implemented in random_forest/
	* generate_meta_data.py generates the metadata
	* classify_emails.py takes the generated metadata, categorises it into training/testing/validation and implements the random forest classifier
	
The K-Means Clustering Algorithm is implemented in k-means_tfidf/
	* top_terms.py scores the importance of words in emails using tfidf
	* plot_top_terms.py uses k-means to plot the top terms

The Naive Bayes Classifier is implemented in bayes/
	* generate_meta_data.py generates the metadata
	* calculate_statistics.py calculates the statistics behind the metadata features
	* classify.py uses the calculated statistics to determine the probability of an email belonging to either human or nonhuman by comparing its metadata attributes to the overall statistics using a Gaussian Probability Density Function

The Parallel Convolutional Neural Network is implemented in parallel_cnn/
	* prepare_data.py finds the most frequently used words and assigns them a score, this score will then be used to vectorise the emails so they can be fed into the neural network
	* classification_network.py implements the neural network and performs a few more modifications on the data

Link to presentation video: https://youtu.be/5gfe4IyNvGg