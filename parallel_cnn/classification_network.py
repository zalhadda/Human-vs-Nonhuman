###############
### IMPORTS ###
###############

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from torchvision.utils import save_image

############
### SEED ###
############

np.random.seed(0)

#################################
### GLOBAL CONSTANT VARIABLES ###
#################################

learning_rate = 0.01
moment = 0.99
lr_decay = 0.75

human_common_words_file = "common_words_human"
nonhuman_common_words_file = "common_words_nonhuman"

num_common_words = 1003 # number of common words

word_limit = 2500 # the word limit per email

possible_outputs = ['nonhuman', 'human']

human_target_output = 1
nonhuman_target_output = 0
border = 0.5

train_batch_size = 3
test_batch_size = 16
epochs = 21

#####################################
### GLOBAL NON-CONSTANT VARIABLES ###
#####################################

# Common Words & Scores

human_words_scores = dict()
nonhuman_words_scores = dict()

# Vectorised Training Data

human_train = list()
nonhuman_train = list()

human_train_padded_numpy = None
nonhuman_train_padded_numpy = None

# Vectorised testing data

human_test = list()
nonhuman_test = list()

human_test_padded_numpy = list()
human_test_padded_numpy = list()

######################
### NEURAL NETWORK ###
######################

'''
Implements a convolutional neural network to classify the emails

    email_length = word_limit
    num_classes = num possible outputs (ex/ human or nonhuman)
    vocab_size = size of the vocaulary
    embedding_dim = dimensionality of the embedding
    num_kernels = num of kernels 
    kernels = kernels as a list
'''

class Email_Classifier(nn.Module):

    def __init__(self, email_length = word_limit, 
                        num_classes = len(possible_outputs), 
                        vocab_size = num_common_words,
                        embedding_dim = 3, 
                        num_kernels = 3, 
                        kernels = [3, 4, 5], 
                        dropout_p = 0.5):
        
        super(Email_Classifier, self).__init__()        

        # Initialise

        self.email_length = email_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_kernels = num_kernels
        self.kernels = kernels
        self.dropout_p = dropout_p

        # Create the embedding layer

        self.word_embedding = nn.Embedding(email_length, embedding_dim)

        # Create the convolution layers
        # We want a different layer for each kernel

        self.conv_layers = nn.ModuleList([ nn.Conv1d(in_channels = 1,
                                                    out_channels = num_kernels,
                                                    kernel_size = kernel,
                                                    stride = embedding_dim)
                                            for kernel in kernels])

    
        # Create the dropout layer
        # Will only be used for training, not testing

        self.dropout = nn.Dropout(dropout_p)

        # Create the fully connected linear layer
        # There will be num_kernels ^ 2 input features
        # The num of output channels is equal to the num_classes (human vs nonhuman)

        self.fully_connected_layer = nn.Linear(num_kernels * len(kernels), num_classes)

    # End of __init__
    

    '''
    Takes in an input_x, assumed to be a vectorised email,
    and runs it through the network.
    '''

    def forward(self, input_x):
        # Apply each conv layer on the input_x

        inputs = [conv(input_x.unsqueeze(1)) for conv in self.conv_layers]

        # Apply a ReLU nonlinearity on each of the generated inputs

        inputs_relu = [F.relu(inp) for inp in inputs]
         
        # Apply max pooling on each input

        inputs_pool = [F.max_pool1d(inp, inp.size(2)).squeeze(2) for inp in inputs_relu]

        # Concatenate all the layers

        input_x = torch.cat(inputs_pool, 1)
        
        # Apply dropout

        input_x = self.dropout(input_x)

        # Connect all the layers

        input_x = self.fully_connected_layer(input_x)

        # Pass the layers through a sigmoidal function

        #print(torch.max(input_x))
        #print(input_x.shape)

        result = torch.sigmoid(input_x)

        return result

    # End of forward()

# End of class Model

##########################
### TRAINING & TESTING ###
##########################

def Create_Training_Batch():

    all_train_data = {}
    iterations = len(human_train_padded_numpy) + len(nonhuman_train_padded_numpy)

    for i in range(max(len(human_train_padded_numpy), len(nonhuman_train_padded_numpy))):
        
        if (i < len(human_train_padded_numpy)):
            all_train_data[len(all_train_data)] = (human_train_padded_numpy[i], human_target_output)

        if (i < len(nonhuman_train_padded_numpy)):
            all_train_data[len(all_train_data)] = (nonhuman_train_padded_numpy[i], nonhuman_target_output)

    # Batched data

    batched_train_data = []
    count = 0
    temp_array = []

    for key, value in all_train_data.items():

        if ((count % train_batch_size) == 0) and (count != 0):
            batched_train_data.append(temp_array)
            temp_array = []

        temp_array.append(value)
        count += 1

    return batched_train_data

# End of Create_Training_Batch


def Train(model):

    train_batch = Create_Training_Batch()

    # Defines the loss function

    loss_function = nn.CrossEntropyLoss()

    # Defines the optimiser

    optimiser = optim.SGD(model.parameters(), lr = learning_rate, momentum = moment)

    # Decay the learning rate by a factor of 0.1 every 7 seconds

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size = 10, gamma = lr_decay)

    # Start Training

    model.train()

    epoch_number = []
    epoch_accuracy = []
    epoch_loss = []

    start_time_epoch = time.time()

    for epoch in range(epochs):

        training_loss = 0
        correct_predictions = 0
        start_time_batch = time.time()

        count = 0
        for batch in train_batch:

            # Retrieves the emails & targets for the current batch

            emails = []
            targets = []

            for (email, target) in batch:
                emails.append(email)
                targets.append(target)

            emails = Variable(torch.from_numpy(np.array(emails)).double())
            targets = Variable(torch.from_numpy(np.array(targets)).double(), requires_grad = False)

            # Clears the gradient

            optimiser.zero_grad()

            # Forward propagation

            output = model(emails.float())

            # Compute loss

            batch_loss = loss_function(output, targets.long())

            # Adds the batch's loss to the overall training_loss

            training_loss += batch_loss.item()

            # Backward propagation

            batch_loss.backward()

            # Update parameters

            optimiser.step()

            # Checks for correct predictions

            value, index = torch.max(output, 1)

            for i in range(train_batch_size):
                result = torch.max(output[i]).item()
                expected = targets[i].item()

                if (result > border and expected == human_target_output) or (result <= border and expected == nonhuman_target_output):
                    correct_predictions += 1

            # Batch-specific statistics

            end_time_batch = time.time() - start_time_batch

            # Print statistics every 50 runs

            if ((count % 50) == 0):
                print("\n----------")
                print("Epoch # " + str(epoch))
                print("Batch # " + str(count))
                print("Batch Loss " + str(batch_loss.item()))
                print("Trained on " + str(((count + 1) * train_batch_size)) + " emails.")
                print("Time Passed: " + str(end_time_batch) + " seconds.")
                print("----------\n")

            count += 1

        # Overall training statistics

        training_accuracy = (correct_predictions / (train_batch_size * len(train_batch))) * 100
        average_training_loss = training_loss / (len(train_batch))
        end_time_epoch = time.time() - start_time_epoch

        train_log("\n##########")
        train_log("Epoch #" + str(epoch))
        train_log("Training Accuracy: " + str(training_accuracy) + "%")
        train_log("Average Training Loss: " + str(average_training_loss))
        train_log("Total Time Taken: " + str(end_time_epoch) + "seconds.")
        train_log("##########\n")

        # Save Statistics

        epoch_number.append(epoch)
        epoch_accuracy.append(training_accuracy)
        epoch_loss.append(average_training_loss)


    # Print overall lists

    print('\n=== epochs ===')
    print(epoch_number)

    print('\n=== epoch accuracies ===')
    print(epoch_accuracy)

    print('\n=== epoch loss ===')
    print(epoch_loss)

    # Save model

    torch.save(model, "model.dat")

# End of Train()

def Create_Testing_Batch():

    all_test_data = {}
    iterations = len(human_test_padded_numpy) + len(nonhuman_test_padded_numpy)

    for i in range(max(len(human_test_padded_numpy), len(nonhuman_test_padded_numpy))):
        
        if (i < len(human_test_padded_numpy)):
            all_test_data[len(all_test_data)] = (human_test_padded_numpy[i], human_target_output)

        if (i < len(nonhuman_train_padded_numpy)):
            all_test_data[len(all_test_data)] = (nonhuman_test_padded_numpy[i], nonhuman_target_output)

    # Batched data

    batched_test_data = []
    count = 0
    temp_array = []

    for key, value in all_test_data.items():

        if ((count % test_batch_size) == 0) and (count != 0):
            batched_test_data.append(temp_array)
            temp_array = []

        temp_array.append(value)
        count += 1

    return batched_test_data

# End of Create_Training_Batch


def Test(model):

    test_batch = Create_Testing_Batch()

    # Defines the loss function

    loss_function = nn.CrossEntropyLoss()

    # Defines the optimiser

    optimiser = optim.SGD(model.parameters(), lr = learning_rate, momentum = moment)

    # Decay the learning rate by a factor of 0.1 every 7 seconds

 #   exp_lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size = 7, gamma = 0.5)

    # Start Testing

    model.eval()

    epoch_number = []
    epoch_accuracy = []
    epoch_loss = []

    start_time_epoch = time.time()

    for epoch in range(epochs):

        testing_loss = 0
        correct_predictions = 0
        start_time_batch = time.time()

        count = 0
        for batch in test_batch:

            # Retrieves the emails & targets for the current batch

            emails = []
            targets = []

            for (email, target) in batch:
                emails.append(email)
                targets.append(target)

            emails = Variable(torch.from_numpy(np.array(emails)).double())
            targets = Variable(torch.from_numpy(np.array(targets)).double(), requires_grad = False)

            # Clears the gradient

            optimiser.zero_grad()

            # Forward propagation

            output = model(emails.float())

            # Compute loss

            batch_loss = loss_function(output, targets.long())

            # Adds the batch's loss to the overall training_loss

            #testing_loss += batch_loss.item()

            # Backward propagation

            #batch_loss.backward()

            # Update parameters

#            optimiser.step()

            # Checks for correct predictions

            value, index = torch.max(output, 1)

            for i in range(test_batch_size):
                result = torch.max(output[i]).item()
                expected = targets[i].item()

                if (result > border and expected == human_target_output) or (result <= border and expected == nonhuman_target_output):
                    correct_predictions += 1

            # Batch-specific statistics

            end_time_batch = time.time() - start_time_batch

            # Print statistics every 50 runs

            if ((count % 50) == 0):
                print("\n----------")
                print("Testing Epoch # " + str(epoch))
                print("Testing Batch # " + str(count))
                print("Testing Batch Loss " + str(batch_loss.item()))
                print("Tested on " + str(((count + 1) * test_batch_size)) + " emails.")
                print("Time Passed: " + str(end_time_batch) + " seconds.")
                print("----------\n")

            count += 1

        # Overall training statistics

        testing_accuracy = (correct_predictions / (test_batch_size * len(test_batch))) * 100
        average_testing_loss = testing_loss / (len(test_batch))
        end_time_epoch = time.time() - start_time_epoch

        test_log("\n##########")
        test_log("Testing Epoch #" + str(epoch))
        test_log("Testing Accuracy: " + str(testing_accuracy) + "%")
        test_log("Average Testing Loss: " + str(average_testing_loss))
        test_log("Total Time Taken: " + str(end_time_epoch) + "seconds.")
        test_log("##########\n")

        # Save Statistics

        epoch_number.append(epoch)
        epoch_accuracy.append(testing_accuracy)
        epoch_loss.append(average_testing_loss)


    # Print overall lists

    print('\n=== testing epochs ===')
    print(epoch_number)

    print('\n=== testing epoch accuracies ===')
    print(epoch_accuracy)

    print('\n=== testing epoch loss ===')
    print(epoch_loss)

# End of Test()

###########
### LOG ###
###########

def train_log(string):
    print(string)

    with open("train_log.txt", "a+") as f:
        f.write(string + '\n')

# End of log

def test_log(string):
    print(string)

    with open("test_log.txt", "a+") as f:
        f.write(string + '\n')

# End of log

########################
### PREPROCESS DATA  ###
########################

'''
Loads the scores of the common words for humans and nonhumans
'''

def Load_Scores():

    # Human Common Words & Scores

    with open(human_common_words_file) as f:
        lines = f.readlines()

        for line in lines:

            # Create tokens

            tokens = line.split()
            word = tokens[0]
            score = tokens[1]

            # Add to global dictionary

            human_words_scores[word] = score



    # Nonhuman Common WOrds & Scores

    with open(nonhuman_common_words_file) as f:
        lines = f.readlines()

        for line in lines:

            # Create tokens

            tokens = line.split()
            word = tokens[0]
            score = tokens[1]

            # Add to global dictionary

            nonhuman_words_scores[word] = score

# End of Load_Scores

'''
Loads the training data that will be used to train the network.
All string values will be vectorised into the common_words scores.
If a word is not a common_word, it will have a value of 0.
START and END tokens will have values of -1 and -2, respectively.

Human training data is expected to be in "training/human/"
Nonhuman training data is expected to be in "training/nonhuman/"
'''

def Load_Training_Data():

    # Human Training Data

    human_training_directory = "training/human/"

    for email in os.listdir(human_training_directory):

        # Open & read the email

        email_path = human_training_directory + email

        with open(email_path, "r") as f:
            lines = f.readlines()

            for line in lines:
    
                # Split the line into words

                words = line.split()

                current_email = list()

                for word in words:
                    
                    # Retrieve the score for each word

                    score = human_words_scores.get(word)

                    # Either append the score, if it exists, or append 0

                    if (score is None):
                        current_email.append(0)
                    else:
                        current_email.append(int(score))

                # Append the scored email to the human training data

                human_train.append(current_email)



    # Nonhuman Training Data

    nonhuman_training_directory = "training/nonhuman/"

    for email in os.listdir(nonhuman_training_directory):

        # Open & read the email

        email_path = nonhuman_training_directory + email

        with open(email_path, "r") as f:
            lines = f.readlines()

            for line in lines:
    
                # Split the line into words

                words = line.split()

                current_email = list()

                for word in words:
                    
                    # Retrieve the score for each word

                    score = nonhuman_words_scores.get(word)

                    # Either append the score, if it exists, or append 0
                    # to the nonhuman training data

                    if (score is None):
                        current_email.append(0)
                    else:
                        current_email.append(int(score))
                
                nonhuman_train.append(current_email)


# End of Load_Training_Data()

'''
Loads testing data
'''

def Load_Testing_Data():

    # Human Training Data

    human_testing_directory = "testing/human/"

    for email in os.listdir(human_testing_directory):

        # Open & read the email

        email_path = human_testing_directory + email

        with open(email_path, "r") as f:
            lines = f.readlines()

            for line in lines:
    
                # Split the line into words

                words = line.split()

                current_email = list()

                for word in words:
                    
                    # Retrieve the score for each word

                    score = human_words_scores.get(word)

                    # Either append the score, if it exists, or append 0

                    if (score is None):
                        current_email.append(0)
                    else:
                        current_email.append(int(score))

                # Append the scored email to the human training data

                human_test.append(current_email)



    # Nonhuman Training Data

    nonhuman_testing_directory = "testing/nonhuman/"

    for email in os.listdir(nonhuman_testing_directory):

        # Open & read the email

        email_path = nonhuman_testing_directory + email

        with open(email_path, "r") as f:
            lines = f.readlines()

            for line in lines:
    
                # Split the line into words

                words = line.split()

                current_email = list()

                for word in words:
                    
                    # Retrieve the score for each word

                    score = nonhuman_words_scores.get(word)

                    # Either append the score, if it exists, or append 0
                    # to the nonhuman training data

                    if (score is None):
                        current_email.append(0)
                    else:
                        current_email.append(int(score))
                
                nonhuman_test.append(current_email)


# End of Load_Training_Data()

'''
Pads all emails in the training data to be of equal length
'''

def Pad_Train_Emails():

    # Human emails

    global human_train_padded_numpy

    count_human_emails = len(human_train)

    # Initialise a numpy array of numpy arrays with length corresponding to # of human emails
    # Each sub-numpy array has length corresponding to the word_limit

    human_train_padded_numpy = np.zeros((count_human_emails, word_limit))

    # Transfer the email word scores from the list() training data to the numpy array

    for email_count in range(count_human_emails):

        for word_index in range(len(human_train[email_count])):
            
            # Keep track of word counts

            if (word_index == word_limit):
                break
           
            # Add the word to the numpy array

            human_train_padded_numpy[email_count][word_index] = human_train[email_count][word_index]
    
    

    # Nonhuman Emails

    count_nonhuman_emails = len(nonhuman_train)

    global nonhuman_train_padded_numpy

    # Initialise a np array of np arrays with length corresponding to # of nonhuman emails
    # Each sub-numpy array has length corresponding to the word_limit

    nonhuman_train_padded_numpy = np.zeros((count_nonhuman_emails, word_limit))

    # Transfer the email word scores from the list() training data to the numpy array

    for email_count in range(count_nonhuman_emails):

        for word_index in range(len(nonhuman_train[email_count])):
            
            # Keep track of word counts

            if (word_index == word_limit):
                break
           
            # Add the word to the numpy array

            nonhuman_train_padded_numpy[email_count][word_index] = nonhuman_train[email_count][word_index]
    

# End of Pad_Emails()

'''
Pads all emails in the testing data to be of equal length
'''

def Pad_Test_Emails():

    # Human emails

    global human_test_padded_numpy

    count_human_emails = len(human_test)

    # Initialise a numpy array of numpy arrays with length corresponding to # of human emails
    # Each sub-numpy array has length corresponding to the word_limit

    human_test_padded_numpy = np.zeros((count_human_emails, word_limit))

    # Transfer the email word scores from the list() training data to the numpy array

    for email_count in range(count_human_emails):

        for word_index in range(len(human_test[email_count])):
            
            # Keep track of word counts

            if (word_index == word_limit):
                break
           
            # Add the word to the numpy array

            human_test_padded_numpy[email_count][word_index] = human_test[email_count][word_index]
    
    

    # Nonhuman Emails

    count_nonhuman_emails = len(nonhuman_test)

    global nonhuman_test_padded_numpy

    # Initialise a np array of np arrays with length corresponding to # of nonhuman emails
    # Each sub-numpy array has length corresponding to the word_limit

    nonhuman_test_padded_numpy = np.zeros((count_nonhuman_emails, word_limit))

    # Transfer the email word scores from the list() training data to the numpy array

    for email_count in range(count_nonhuman_emails):

        for word_index in range(len(nonhuman_test[email_count])):
            
            # Keep track of word counts

            if (word_index == word_limit):
                break
           
            # Add the word to the numpy array

            nonhuman_test_padded_numpy[email_count][word_index] = nonhuman_test[email_count][word_index]
    

# End of Pad_Emails()

############
### MAIN ###
############

def Main():
    
    print("Program Starting...\n")

    # Load scores from the common words files

    Load_Scores()

    print("Scores successfully loaded\n")

    # Load training data
    # Training data is expected to be in a "training" folder in the current directory
    # Human data is expected to be a in a "human" subdirectory of "training"
    # Nonhuman data is expected to be in a "nonhuman" subdirectory of "training"

    Load_Training_Data()

    print("Training data successfully loaded\n")

    # Pad all sentences in the emails to be of equal size

    Pad_Train_Emails()

    print("Training Emails successfully padded\n")
    
    # Load testing data

    Load_Testing_Data()

    print("Testing emails successfully loaded\n")

    # Pad testing emails

    Pad_Test_Emails()

    print("Testing emails successfully padded\n")

    # Initialise Model

    email_classifier_model = Email_Classifier()

    # Train Model

    Train(email_classifier_model)

    # Test Model

    Test(email_classifier_model)

# End of Main()

Main()
