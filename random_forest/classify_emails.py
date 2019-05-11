###############
### IMPORTS ###
###############

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

import sys

############
### SEED ###
############

np.random.seed(0)

#######################
### REMODEL DATA ###
#######################

'''
Takes the metadata and labels file and converts the data into a
Pandas DataFrame format, which will be used in the Random Forest
'''

def To_Pandas_Data_Frame(metadata_file):

    # Reads metadata_file
    # The metadata file has the following format:
    #   Email_ID, Lines, Characters, Letters, Digits, Specials
    # with the first line being the column names
    
    metadata_lines = None

    with open(metadata_file) as f:
        metadata_lines = f.readlines()


    # Organises the contents of the metadata file into its respective arrays

    column_names = []

    id_arr = []
    lines_arr = []
    words_arr = []
    characters_arr = []
    letters_arr = []
    digits_arr = []
    specials_arr = []
    cases_arr = []

    first_line = True

    for line in metadata_lines:
        
        # Removes the \n from the end of the line

        line = line.rstrip('\n')

        # Splits the line into entries

        entries = line.split(',')

        # If it's the first line,
        # then it's the column names

        if first_line:
            
            skip = True

            for entry in entries:
                
                # Don't add the email_ids to the column_names
                # email_ids will be used as the index

                if skip:
                    skip = False
                    continue

                column_names.append(entry.strip())

            first_line = False
            continue


        # Places the entries in the respective array
        # Removes the whitespace before/after the entries

        id_arr.append(entries[0].strip())
        lines_arr.append(entries[1].strip())
        words_arr.append(entries[2].strip())
        characters_arr.append(entries[3].strip())
        letters_arr.append(entries[4].strip())
        digits_arr.append(entries[5].strip())
        specials_arr.append(entries[6].strip())
        cases_arr.append(entries[7].strip())

    '''

    # Reads the labels_file
    # The kabels_file ahs the following format:
    #   Email_ID, Case
    # with the first line being the column names
    # and Case denoting whether an email is a bot ('0') or a human ('1') email

    labels_lines = None

    with open(labels_file) as f:
        labels_lines = f.readlines()

    # Organises the cases_arr
    # This array holds the classification of the respective email

    cases_arr = [None] * len(id_arr)

    first_line = True

    for line in labels_lines:
        # Removes the \n from the end of the line

        line = line.rstrip('\n')

        # Splits the line into entries

        entries = line.split(',')
        
        # If it's the first line, then it's the column names
        # We only care about the 'Case' column name

        if first_line:
            
            column_names.append(entries[1].strip())

            first_line = False
            continue

        # The first entry, corresponding to the email_id, will be used to find
        # the array index of the email_id in id_arr.
        # The second entry, the case, will be placed in the case_arr corresponding
        # to the index found.
        # This is used to maintain the order of the data

        index = id_arr.index(entries[0].strip())
        cases_arr[index] = entries[1].strip()

    '''

    # Aggregates the data

    data = {column_names[0]: lines_arr,
            column_names[1]: words_arr,
            column_names[2]: characters_arr,
            column_names[3]: letters_arr,
            column_names[4]: digits_arr,
            column_names[5]: specials_arr,
            column_names[6]: cases_arr}

    # Creates the Pandas Data Frame

    df = pd.DataFrame(data, columns = column_names)

    # Sets the index of the Data Frame to the id_arr

    df.index = id_arr

    return df

# End of To_Pandas_Data_Frame()

'''
Creates the Training, Testing, and Validation datasets from the given Pandas Data Frame
    80% of the data will be Training Data
    10% of the data will be Testing Data
    10 % of the data will be Validation Data
'''

def Create_Train_Test_Validate_Data(df):
    
    # Adds the is_train column
    # 80% of the data is assigned randomly assigned to be the training data
    # The remaining 20% of the data will be used as the testing/validation data

    df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.80

    train = df[df['is_train'] == True]
    not_train = df[df['is_train'] == False]

    # Deletes the 'is_train' column

    train = train.drop('is_train', 1)
    not_train = not_train.drop('is_train', 1)

    # Adds the is_test column to the not_train column
    # 50% of the data will randomly be assigned to be the testing data
    # The other 50% of the data will be the validation data

    not_train['is_test'] = np.random.uniform(0, 1, len(not_train)) <= 0.50

    test = not_train[not_train['is_test'] == True]
    validate = not_train[not_train['is_test'] == False]

    # Deletes the 'is_test' column

    test = test.drop('is_test', 1)
    validate = validate.drop('is_test', 1)

    return train, test, validate

# End of Create_Train_Test_Validate()

##################
### STATISTICS ###
##################

'''
Prints the number of entires in each dataset
'''

def Print_Counts(train, test, validate):

    if train is not None:
        print("Number of entries in the training dataset: " + str(len(train)))

    if test is not None:
        print("Number of entries in the testing dataset: " + str(len(test)))

    if validate is not None:
        print("Number of entries in the validation dataset: " + str(len(validate)))

# End of Print_Counts()

'''
Determines the test accuracy of the classifier:
    * determined: what the classifier achieved
    * expectd: what was expected
'''

def Test_Accuracy(determined_results, expected_results):

    # Finds the number of matching results

    matching_result = 0

    for i in range(0, len(determined_results)):

        if (int(determined_results[i]) == int(expected_results[i])):
            matching_result += 1

    # Determines the accuracy

    accuracy = matching_result / len(determined_results) * 100

    return accuracy

# End of Test_Accuracy()

################################
### RANDOM FOREST CLASSIFIER ###
################################

'''
Trains the Random Forest
'''

def Train(train, attributes):

    # Creates the random classifier

    classifier = RandomForestClassifier(n_estimators = 10, n_jobs = 2, random_state = 0)

    # Trains the classifier

    classifier.fit(train[attributes], train['Case'])

    return classifier

# End of Train()

'''
Tests the trained classifier
'''

def Test(classifier, test, attributes):

    test_results = classifier.predict(test[attributes])

    return test_results

# End of Test()

############
### MAIN ###
############

def Main():

    # Check for appropriate number of command line arguments

    if (len(sys.argv) != 2):
        print("Usage: classify_emails.py <path/to/metadata>.\n")
        sys.exit()

    # Retrieve the second argument
    # This argument is expected to be the metadata file

    metadata_arg = str(sys.argv[1])

    # Retrieve the third argument
    # This argument is expected to be the labels file

    #labels_arg = str(sys.argv[2])

    # Check that both arguments point to a valid file

    metadata_file = Path(metadata_arg)
    #labels_file = Path(labels_arg)

    if not (metadata_file.is_file()):
        print("Error: one or more of the arguments do not point to valid file(s).")
        sys.exit()

    # Convert the metadata and labels files contents to a Pandas Data Frame

    df = To_Pandas_Data_Frame(metadata_file)
    
    # Retrieves the attributes of the data
    # i.e. all the columns except for the Case, the last column

    attributes = df.columns[:-1]

    # Creates the Training dataset

    train_df, test_df, validate_df = Create_Train_Test_Validate_Data(df)

    # Prints current stats of the training, testing, and validation datasets

    Print_Counts(train_df, test_df, validate_df)

    # Train the classifier

    print(attributes)

    classifier = Train(train_df, attributes)

    # Tests the classifier

    test_results = Test(classifier, test_df, attributes)

    # Determine the testing accuracy

    test_targets = pd.factorize(test_df['Case'])[0]

    expected_results = []

    for target in test_targets:
        expected_results.append(target)

    accuracy = Test_Accuracy(test_results, expected_results) 

    print("Testing accuracy: " + str(accuracy) + "%") 

# End of Main

Main()

