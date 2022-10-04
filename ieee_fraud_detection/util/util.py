import zipfile
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

# Function to load data:
def zipped_csv_to_df(path='', file_name='', cols=None):
    
    '''Function to open .csv zipped files. Outputs pandas data frame.'''
    
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[file_name]
        data = archive.read(name)
        data = pd.read_csv(BytesIO(data), parse_dates=['timestamp'], usecols=cols)
    return data