###############
### IMPORTS ###
###############

from pathlib import Path
from shutil import rmtree

import numpy as np

import nltk
import os
import sys

############
### SEED ###
############

np.random.seed(0)

####################
### REMODEL TEXT ###
####################

'''
Tokenises the text in a given file,
Removes the infrequent words,
and prepends/appends start/end tokens
'''

def Read_And_Modify(filepath):

    email_body = ""

    # Opens the given email

    with open(filepath) as email:
        lines = email.readlines()

        for line in lines:
            email_body = email_body + line

    # Tokenise the sentences

    sentences = nltk.tokenize.sent_tokenize(email_body)

    # Append & Prepend Start/End tokens to each sentence
    #   START: "SENTENCE_START"
    #   END: "SENTENCE_END"

#    sentences = ["%s %s %s" %("SENTENCE_START", sentence, "SENTENCE_END") for sentence in sentences]

    # Tokenise each sentence into words

    sentences_words = list()

    for sentence in sentences:
        sentences_words.append(nltk.tokenize.word_tokenize(sentence))

    # Generate remodelled email

    new_email = ""

    for sentence in sentences_words:

        for word in sentence:

            new_email = new_email + " " + word

    # Remove start and ending spaces

    new_email.strip()

    return new_email

# End of Read()

###############################
### TRAIN & VALIDATE & TEST ###
###############################

'''
Determines whether to add the email to the training, validation, or testing datasets.
There's an 80% chance the email is added to training.
There's a 10% chance the email is added to validation.
There's a 10% chance the email is added to testing.
'''

def Determine_Dataset():

    # 80% chance to be added to training/

    if not (np.random.uniform(0, 1) >= 0.80):
        return "training/"

    # 10% chance to be added to either validation/ or testing/

    if (np.random.uniform(0, 1) >= 0.50):
        return "validation/"
    else:
        return "testing/"

# End of Determine_Dataset()

############
### MAIN ###
############

def Main():

    # The second argument is expected to be the directory of the emails

    email_directory = str(sys.argv[1])

    # The third argument defines whether the emails are human or nonhuman
    #   This value will either be "human" or "nonhuman"

    classification = str(sys.argv[2])

    # Retrieve each file in the email_directory

    training_count = 0
    validation_count = 0
    testing_count = 0

    email_count = 0

    all_emails = "" # Groups all emails sent to training/
                    # They will be used to find the most frequently-used words

    for filename in os.listdir(email_directory):
        
        email_count += 1
        
        # Create the file_path

        file_path = email_directory + filename
        
        # Tokenise text, remove infrequent words, and preprend start/end tokens

        email = Read_And_Modify(file_path)

        # Decides where to save the formatted email
        
        output_directory = Determine_Dataset()

        # Keep track of statistics

        if (output_directory == "training/"):
            training_count += 1
            all_emails = all_emails + email # To find most frequently-used words
        
        elif (output_directory == "validation/"):
            validation_count += 1
        
        elif (output_directory == "testing/"):
            testing_count += 1

        # Save the email

        email_file_name = classification + "_" + str(email_count)
        email_file_path = output_directory + email_file_name

        with open(email_file_path, "w") as email_file:
            email_file.write(email)

    # Statistics

    statistics_file = "statistics_" + classification

    with open(statistics_file, "w") as f:
        f.write("training_count: " + str(training_count) + "\n")
        f.write("validation_count: " + str(validation_count) + "\n")
        f.write("testing_count: " + str(testing_count) + "\n")
        f.write("total_count: " + str(training_count + testing_count + validation_count) + "\n")

    # Find most frequently used words

    word_frequency = nltk.FreqDist(nltk.tokenize.word_tokenize(all_emails))

    # Take the top 1002 most common words
    #   2 extra to account for START/END tokens

    common_words = word_frequency.most_common(1002)

    words_in_index_order = [word_freq[0] for word_freq in common_words]
    
    # Write the words to a file and assign them a score from 1000 (common) to 1 (uncommon)

    common_words_file = "common_words_" + classification

    with open(common_words_file, "w") as f:
    
        score = 1000

        for word in words_in_index_order:

            if (word == "SENTENCE_START" or word == "SENTENCE_END"):
                continue

            f.write(word + " " + str(score) + "\n")
            score -= 1

        f.write("SENTENCE_START -1\n")
        f.write("SENTENCE_END -2\n")

# End of Main()

Main()
