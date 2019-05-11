###############
### IMPORTS ###
###############

from pathlib import Path
from shutil import rmtree

import os
import re
import sys

########################
### GLOBAL VARIABLES ###
########################

meta_data_file_name = None
email_id = 0

########################
### META DATA OUTPUT ###
########################

'''
Creates the meta data output file
The file will be named "<file_name>_meta_data.txt"
where <file_name> is passed as a parameter
'''

def Create_Meta_Data_File(classification):

    # Generates the output file name

    global meta_data_file_name 
    meta_data_file_name = str(classification) + "_meta_data.txt"

    # If the file already exists, removes it
    
    if (os.path.exists(meta_data_file_name)):
        os.remove(meta_data_file_name)

    # Creates the file with write permissions

    meta_data_file = open(meta_data_file_name, "w+")

    # Writes the meta data labels as the first line
    
#    meta_data_file.write("Email_ID, Lines, Words, Characters, Letters, Digits, Specials, Case\r\n")

    # Closes the file

    meta_data_file.close()

# End of Create_Meta_Data_File

#######################
### WRITE TO OUTPIT ###
#######################

'''
Appends the given string to the meta data output file
'''

def Write_To_Output(string):

    with open(meta_data_file_name, "a") as f:
        f.write(string)

# End of Write_To_Output()

##################
### SAVE EMAIL ###
##################

'''
Saves the source email in its own file named email_<id #>
'''

def Save_Email(email_id, body):

    file_name = "email_" + str(email_id)
    file_path = "emails/" + file_name

    with open(file_path, "w") as f:
        f.write(body)

# End of Save_Email

#################
### READ FILE ###
#################

'''
Reads the contents of a given file,
and extracts the following information:
    - Number of letters
        - Number of upper-case letters
        - Number of lower-case letters
    - Number of digits
    - Number of special characters, and
    - Number of words
    - Total number of characters
'''

def Read(file_name, classification):

    lines = None

    # Opens the file and stores all the lines

    with open(file_name) as f:
        lines = f.readlines()

    # Initialises the counts

    line_count = 0
    word_count = 0
    character_count = 0
    letter_count = 0
    digits_count = 0
    specials_count = 0

    # This will store the email body so it can be saved later

    email_body = ""

    # Reads the email bodies

    for line in lines:

        # Resets the stats
        # and moves to the next line

        #if ("start_of_body" in line):

        #    line_count = 0
        #    word_count = 0
        #    character_count = 0
        #    letter_count = 0
        #    digits_count = 0
        #    specials_count = 0

        #    email_body = ""

         #   continue

        # Writes the stats to the meta data output file
        # and moves to the next line

        #if ("end_of_body" in line):
            
        #    # Dynamically generates the email id

         #   global email_id
         #   email_id += 1

            # If the email has no content, skip it

         #   if (line_count == 0):
         #       continue

            # Sets up the string

            
            # Saves the email in its own file

            #Save_Email(email_id, email_body)

            # Move onto the next line

            #continue
        
        # Saves the lines

        email_body = email_body + line

        # Matches letters

        letters_only = re.findall('[a-zA-Z]+', line)

        # Matches digits
        
        digits_only = re.findall('[0-9]+', line)

        # Matches special characters

        specials_only = re.findall('[\W]+', line)

        # Splits the line into words

        words = line.split()

        # Updates meta data stats for the current body

        line_count += 1
        word_count += len(words)
        character_count += len(line)
        letter_count += len(letters_only)
        digits_count += len(digits_only)
        specials_count += len(specials_only)

    global email_id
    email_id += 1

    stats = (classification + "_" + "email_" + str(email_id) + ", "
                    + str(line_count) + ", "
                    + str(word_count) + ", " 
                    + str(character_count) + ", "
                    + str(letter_count) + ", " 
                    + str(digits_count) + ", " 
                    + str(specials_count) + ", "
                    + str(1 if (classification == "human") else 0) + "\r\n")

    # Writes the string to the output

    Write_To_Output(stats)
    

# End of Read()

############
### MAIN ###
############

def Main():
    # Check for appropriate number of command-line arguments

    if (len(sys.argv) != 3):
        print ("Usage: generate_meta_data.py <path/to/emails> <human/nonhuman>.\n")
        sys.exit()

    

    # Retrieves the second command-line argument
    # This argument is expected to be the email_directory

    email_directory = str(sys.argv[1])

    # Retrieves the email classifications

    classification = str(sys.argv[2])

    # Creates the meta data output file

    Create_Meta_Data_File(classification)

    # Iterates through all the emails in the directory

    for filename in os.listdir(email_directory):

        Read(email_directory + filename, classification)

    '''

    # Check if cmd_line_file_path points to a file

    my_file = Path(cmd_line_file_path)

    if not (my_file.is_file()):
        print("Error: <" + cmd_line_file_path + "> does not point to a valid file.")
        sys.exit()

    # Creates a directory to store all the emails
    # If it already exists, deletes it

    if os.path.exists("emails"):
        rmtree("emails")

    os.makedirs("emails"

    # Creates the meta data output file

    Create_Meta_Data_File(my_file)

    # Reads the content of the file to retrieve the meta data

    Read(my_file)
    
    '''

# End of Main()

Main()
