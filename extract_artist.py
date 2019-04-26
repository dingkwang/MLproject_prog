import os
import csv
import sys
#import librosa
import matplotlib
import numpy as np
#import librosa.display
import IPython.display as ipd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from eyed3 import id3
tag = id3.Tag()

# Import other project files
from read_input import get_non_prog_files,get_prog_files

prog_files = get_prog_files()
non_prog_files = get_non_prog_files()

print("Number of prog songs",len(prog_files))
print("Number of non prog songs",len(non_prog_files))

all_files = prog_files + non_prog_files

#------------------------- Feature extraction ----------------------------------------------------
file = open('data_features_artist.csv', 'w', newline='')
header = 'filename genre artist'

#for i in range(1, 21):
#    header += f' mfcc{i}'

header = header.split()

# Create file to write error logs
#error_logs = open("error_logs.txt","w")
#error_logs.close()    
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genre = 'prog'   

# Read prog files 
for i in range(len(prog_files)) :
        print(i)
        filename = prog_files[i]
#        try:
#            y, sr = librosa.load(filename) 
#        except Exception as e :
#            print("error handled")
#            error_logs = open("error_logs.txt","a")
#            error_logs.write(filename)
#            error_logs.write("\n")
#            error_logs.write(str(e))
#            error_logs.write("\n")    
#            error_logs.close()    
#            continue
        tag.parse(filename)
        artist = tag.artist
        if type(artist) is str:
            artist = artist.replace(" ","_")
            
        filename = filename.split("/")
        filename = filename[-1]
        filename = filename.replace(" ","_")
#
#        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#        rmse = librosa.feature.rmse(y=y)
#        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#        zcr = librosa.feature.zero_crossing_rate(y)/len(y)
#        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {genre} {artist}'    


        # Append all mfcc features i.e., 20 rows
#        for e in mfcc:
#            to_append += f' {np.mean(e)}'
        
        file = open('data_features_artist.csv', 'a', newline='', encoding='utf-8')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
      
genre = 'non_prog'   
print(genre)

# Read Non prog files
for i in range(len(non_prog_files)) :
        print(i)
        filename = non_prog_files[i]
#        try:
#            y, sr = librosa.load(filename)
#        except Exception as e:
#            print("error handled")
#            error_logs = open("error_logs.txt","a")
#            error_logs.write(filename)
#            error_logs.write("\n")
#            error_logs.write(str(e))
#            error_logs.write("\n")    
#            error_logs.close()    
#            continue
        tag.parse(filename)
        artist = tag.artist
        if type(artist) is str:
            artist = artist.replace(" ","_")
            
        filename = filename.split("/")
        filename = filename[-1]
        filename = filename.replace(" ","_")
#        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#        rmse = librosa.feature.rmse(y=y)
#        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#        zcr = librosa.feature.zero_crossing_rate(y)
#        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {genre} {artist}'  
       
#        for e in mfcc:
#            to_append += f' {np.mean(e)}'
        
        file = open('data_features_artist.csv', 'a', newline='', encoding='utf-8')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


# Display Spectrogram
# STFT converts signal such that we can know the amplitude of given frequency at a given time. 
# Using STFT we can determine the amplitude of various frequencies playing at a given time of an audio signal
'''
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
    plt.colorbar()
    plt.show()
'''
