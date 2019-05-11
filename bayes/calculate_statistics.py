###############
### IMPORTS ###
###############

from pathlib import Path

import math
import os
import sys

############################
### Calculate Statistics ###
############################

'''
Calculates the mean of a given list of data
'''

def Mean(data):

    return sum(data) / float(len(data))

# End of Mean()

'''
Calculates the stddev of a given list of stddev
'''

def Stddev(data):

    mean = Mean(data)
    variance = sum([pow(x - mean, 2) for x in data]) / float(len(data) - 1)
    return math.sqrt(variance)

# End of Stddev()

############
### MAIN ###
############

def Main():

    # Check for appropriate number of arguments

    if (len(sys.argv) != 3):
        print("Usage: calculate_statistics.py <metadata> <classification>.\n")
        sys.exit()

    # Retrieve the metadata file

    metadata_file = str(sys.argv[1])

    # Retrieve classification

    classification = str(sys.argv[2])

    # TODO: Create the output file

    output_file = classification + "_statistics"

    # If the file already exists, remove it

    if (os.path.exists(output_file)):
        os.remove(output_file)

    output_file = open(output_file, "w+")

    # Read metadata file
    # The file will be in the following format:
    #   email_id, lines, words, characters, letters, digits, specials, case

    # Retrieve the number of lines, words, characters, letters, digits, and special characters
    
    lines_arr = []
    words_arr = []
    characters_arr = []
    letters_arr = []
    digits_arr = []
    specials_arr = []
    
    num_of_samples = -1

    with open(metadata_file) as f:
        lines = f.readlines()

        # Number of lines represent our sample size

        num_of_samples = len(lines)

        # Retrieve the attribute entries

        for line in lines:

            # Remove the \n from the end of line

            line = line.rstrip('\n')

            # Tokenise the line

            tokens = line.split(',')

            words_arr.append(int(tokens[1].strip()))
            characters_arr.append(int(tokens[2].strip()))
            letters_arr.append(int(tokens[3].strip()))
            digits_arr.append(int(tokens[4].strip()))
            specials_arr.append(int(tokens[5].strip()))

    # Aggregates data

    data = {"words": words_arr, 
            "characters": characters_arr, 
            "letters": letters_arr, 
            "digits": digits_arr, 
            "specials": specials_arr}

    # Calculate the mean and standard deviation per attribute

    for feature, attribute in data.items():

        mean = str(Mean(attribute))
        stddev = str(Stddev(attribute))

        print("\n---" + feature + "---")
        print("Mean:", mean)
        print("Stddev:", stddev)

        output_file.write(feature + ": " + str(num_of_samples) + ", " + mean + ", " + stddev + "\r\n")


    output_file.close()

# End of Main()

Main()
