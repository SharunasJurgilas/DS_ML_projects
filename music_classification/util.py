import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import librosa
import soundfile as sf
import io
import IPython
from scipy import signal


genres = ['blues', 'classical', 'country', 
          'disco', 'hiphop', 'jazz', 'metal', 
          'pop', 'reggae', 'rock']

def build_catalogue():
    catalogue = {}
    with zipfile.ZipFile('C:/Users/sjurg/OneDrive/Documents/Python/misc_data_sets/music-genres/archive.zip') as archive:
        names = archive.namelist()
        for genre in genres:
            songs = [name for name in names if name.startswith('Data/genres_original/' + genre)]
            catalogue.update({genre:songs})
    return catalogue

def build_library(catalogue, c_type = 'dict', sr = True):
    if c_type == 'dict':
        library = {}
    else:
        library = []
    for genre in genres:
        with zipfile.ZipFile('C:/Users/sjurg/OneDrive/Documents/Python/misc_data_sets/music-genres/archive.zip') as archive:
            songs = []
            for song in catalogue[genre]:
                with archive.open(song) as myfile:
                    tmp = io.BytesIO(myfile.read())
                    try:
                        data, samplerate = sf.read(tmp)
                    except:
                        pass
                songs.append(data)
            if c_type == 'dict':
                library.update({genre:songs})
            else:
                library.append(songs)
    if sr:
        print('Sample rate is:{}'.format(samplerate))
    return library