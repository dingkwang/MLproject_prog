import numpy as np
import os
import sys

#current_working_directory = os.getcwd() # get current working directory

current_working_directory = "Z:/Downloads/cap6610sp19_project/"
#current_working_directory = 'C:\Users\Dingkang\OneDrive - University of Florida\MLproject_prog\cap6610sp19_project'
# Address of root directory
base_path = current_working_directory + "/Validation_Set/"
prog_path = base_path + "Prog/"
non_prog_path = base_path + "Non-Prog/"


# Returns all audio files(.mp3, .avi, .wav) in the directory : path
def fileList(path) :
    matches = []
    for root, _, filenames in os.walk(path,topdown=True):
        for filename in filenames:
            if filename.endswith(('.mp3', '.wav', '.avi')):
                if matches.count(filename) == 0 :
                    matches.append(os.path.join(root, filename))
            
    return matches


# prog_files contains all prog_rock files
prog_files = fileList(prog_path)

# non_prog_files contains all non prog_rock files
non_prog_files = fileList(non_prog_path)


# Returns all prog rock files
def get_prog_files_r():
    return prog_files


# Returns all non prog rock files
def get_non_prog_files_r() :
    return non_prog_files    
