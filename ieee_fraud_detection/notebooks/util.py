import zipfile
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

# Function to load data:
def zipped_csv_to_df(path='', file_name='', cols=None):
    
    '''Function to open .csv zipped files. Outputs pandas data frame.'''
    
    with zipfile.ZipFile(path) as archive:
        #name = archive.namelist()[file_name]
        data = archive.read(file_name)
        data = pd.read_csv(BytesIO(data), usecols=cols)
    return data

def get_low_missing_features(data, thr):
    '''Function which returns a list of features from the data dataframe,
    which have less than thr missing features'''
    missing_val_counts = data.isna().sum()
    return missing_val_counts[missing_val_counts < thr].index.tolist()

def reduce_data_size(data):
    '''
    Converts all float and integer types to appropriate types based on the range of values.
    Helps to substantially reduce the memory usage of a data frame.
    Data types considered:
    int8;
    int16;
    int32;
    int64;
    float16;
    float32;
    float64.
    Object types left unchanged.
    '''
    for c in data.columns:
        c_type = data[c].dtype
        
        if c_type != object and str(c_type).startswith('int'):
            cmin = data[c].min()
            cmax = data[c].max()
            
            if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                data[c] = data[c].astype(np.int8)
                
            elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                data[c] = data[c].astype(np.int16)
            
            elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                data[c] = data[c].astype(np.int32)
                
            elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                data[c] = data[c].astype(np.int64)
                
        elif c_type != object and str(c_type).startswith('float'):
            cmin = data[c].min()
            cmax = data[c].max()
            
            if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                data[c] = data[c].astype(np.float16)
                
            elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                data[c] = data[c].astype(np.float32)
                
            elif cmin > np.finfo(np.float64).min and cmax < np.finfo(np.float64).max:
                data[c] = data[c].astype(np.float64)
                
    return data

def ordinal_encoding(data):
    '''
    Finds all cartegorical features and does ordinal encoding.
    '''
    categorical_features = data.select_dtypes(include=['object']).columns
    numerical_features = data.select_dtypes(include=['number', 'datetime']).columns
    for feature in categorical_features:
        data[feature] = pd.factorize(data[feature])[0]
    return data