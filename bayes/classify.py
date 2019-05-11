###############
### IMPORTS ###
###############

import math
import os
import re
import sys

########################
### GLOBAL VARIABLES ###
########################

summary = dict()

######################
### MAKE SUMMARIES ###
######################

'''
Retrieve the statistics from a given file.
Files are expected to be in the following format:
    attribute: # of samples, mean, stddev
'''

def Statistical_Summary(statistics_file):

    # Retrieve statistics

    statistics = []

    with open(statistics_file) as f:
        lines = f.readlines()

        for line in lines:
            tokens = line.split()

            # tokens[2] and tokes[3] are the mean and stddev for the given file

            statistics.append((float(tokens[2].split(',')[0]), float(tokens[3])))

    return statistics

# End of Statistical_Summary()

#######################
### MAKE PREDICTION ###
#######################

'''
Calculates the Gaussian Probability Denisty Function of a given attribute
'''

def Calculate_GPDF(attribute_value, attribute_mean, attribute_stddev):

    # Calculate exponent

    exponent = math.exp(-(math.pow(attribute_value - attribute_mean, 2) / (2 * math.pow(attribute_stddev, 2))))

    # Perform division

    division = 1 / (math.sqrt(2 * math.pi) * attribute_stddev)

    # Return the gaussian probability density function

    return division * exponent

# End if Calculate_GPDF()

##################
### Test Model ###
##################

'''
Classifies the given email.
    filepath = path of test data
    classification = expected classification of test data
Returns the prediction accuracy.
'''

def Classify_Emails(filepath, classification):

    correct_predictions = 0
    num_samples = 0

    for filename in os.listdir(filepath):

        # Keeps track of # of data tested on

        num_samples += 1

        # Creates the email_path

        email_path = filepath + filename

        # Instantiates data

        #num_lines = 0
        num_words = 0
        num_characters = 0
        num_letters = 0
        num_digits = 0
        num_specials = 0

        with open(email_path) as email:
            lines = email.readlines()

            for line in lines:

                # Isolate all the letters, digits, and special characters in the line

                letters = re.findall('[a-zA-Z]+', line)
                digits = re.findall('[0-9]+', line)
                specials = re.findall('[\W]+', line)

                # Split the line into words

                words = line.split()

                # Update counts

                #num_lines += 1
                num_words += len(words)
                num_characters += len(line)
                num_letters += len(letters)
                num_digits += len(digits)
                num_specials += len(specials)

            # Aggregate the values

            values = []
            #values.append(num_lines)        # 0
            values.append(num_words)        # 1
            values.append(num_characters)   # 2
            values.append(num_letters)      # 3
            values.append(num_digits)       # 4
            values.append(num_specials)     # 5

            # These will be used to store the prediction outcomes

            human_probabilities = []
            nonhuman_probabilities = []

            # Retrieve probabilities

            # Human Probabilities

            current_value = 0

            for statistics in summary['human']:
                mean = float(statistics[0])
                stddev = float(statistics[1])

                probability = Calculate_GPDF(values[current_value], mean, stddev)
                
                human_probabilities.append(probability)
            
            # Nonhuman Probabilities

            current_value = 0

            for statistics in summary['nonhuman']:
                mean = float(statistics[0])
                stddev = float(statistics[1])

                probability = Calculate_GPDF(values[current_value], mean, stddev)
                
                nonhuman_probabilities.append(probability)

            # Make Prediction

            human_score = 0
            nonhuman_score = 0

            for i in range(len(human_probabilities)):

                #print('')

                #print(human_probabilities[i])
                #print(nonhuman_probabilities[i])

                if (human_probabilities[i] > nonhuman_probabilities[i]):
                    human_score += 1
                else:
                    nonhuman_score += 1

     #       print('')
     #       print(human_score)
     #       print(nonhuman_score)
     #       print('')

            # Calculate accuracy

            if (human_score > nonhuman_score) and (classification == 1):
                correct_predictions += 1

            elif (human_score < nonhuman_score) and (classification == 0):
                correct_predictions += 1

    return (correct_predictions / num_samples) * 100

# End of Classify_Emails

'''
Reads the testing data emails and classifies them.
The testing data is expected to be found in:
    testing/human/
    testing/nonhuman/
'''

def Test_Model():

    print("\nTesting on human data...")
    human_accuracy = Classify_Emails("testing/human/", 1)
    print("human_accuracy: " + str(human_accuracy) + " %")

    print("\nTesting on nonhuman data...")
    nonhuman_accuracy = Classify_Emails("testing/nonhuman/", 0)
    print("nonhuman_accuracy: " + str(nonhuman_accuracy) + " %")

    print("\nOverall Accuracy:")
    print(str((human_accuracy + nonhuman_accuracy) / 2))

# End of Test_Model()

############
### MAIN ###
############

def Main():

    # Get statistics files

    human_statistics_file = sys.argv[1]
    nonhuman_statistics_file = sys.argv[2]

    # Retrieve statistics summaries from the respective files
    # Attributes are organises as follows:
    #   lines, words, characters, letters, digits, and specials

    human_statistics = Statistical_Summary(human_statistics_file)
    nonhuman_statistics = Statistical_Summary(nonhuman_statistics_file)

    global summary

    summary['nonhuman'] = nonhuman_statistics
    summary['human'] = human_statistics

    # Loads the test data and makes predictions
    # Returns the accuracy of the model

    Test_Model()

# End of Main()

Main()
