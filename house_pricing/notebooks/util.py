import zipfile
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

# Function to load data:
def zipped_csv_to_df(file_name='', cols=None):
    
    '''Function to open .csv zipped files. Outputs pandas data frame.'''
    
    with zipfile.ZipFile(file_name) as archive:
        name = archive.namelist()[0]
        data = archive.read(name)
        data = pd.read_csv(BytesIO(data), parse_dates=['timestamp'], usecols=cols)
    return data

def get_basic_feature_stats(data):
    
    '''Takes in data as a pandas series object. For example, data_frame['column_name'].
    Returns histogram, mean value, standard deviation, median value min/max values.'''
    
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.hist(data, bins=40, color='teal', ec='black', alpha=0.4)
    ax.set_xlabel(data.name)
    ax.set_ylabel('Counts')
    print('Mean: {}'.format(data.mean()))
    print('Standard deviation: {}'.format(data.std()))
    print('Median: {}'.format(data.median()))
    print('Min value: {}'.format(data.min()))
    print('Max value: {}'.format(data.max()))
    print('Number of data points: {}'.format((~data.isna()).sum()))
    
# Ordinal encoding of categorical features:
def ordinal_encoding(data):
    
    '''Finds all cartegorical features and does ordinal encoding.'''
    
    categorical_features = data.select_dtypes(include=['object']).columns
    numerical_features = data.select_dtypes(include=['number', 'datetime']).columns
    for feature in categorical_features:
        data[feature] = pd.factorize(data[feature])[0]
    return data

# Data wrangling functions. Mostly self explanatory...:
def clean_train_set_only(data):
    data['full_sq'] = data['full_sq'].mask(data['full_sq'] == 0.0, data['full_sq'].median())
    return data

def clean_level_1(data):
    data['build_year'] = (data['build_year'].mask(data['build_year'] < 1500, data['build_year'].median())
                     .mask(data['build_year'] > 2020, data['build_year'].median())
                    )
    data['full_sq'] = data['full_sq'].mask(data['full_sq'] > 210, data['full_sq'].median())
    data['life_sq'] = data['life_sq'].mask(data['life_sq'] > 200, data['life_sq'].median())
    data['full_sq'] = data['full_sq'].mask(data['full_sq'] == 0, 50)
    data['floor'] = data['floor'].mask(data['floor'] > 50, data['floor'].median())
    data['floor'] = data['floor'].mask(data['floor'] > data['max_floor'], data['max_floor'])
    data['life_sq'] = data['life_sq'].mask(data['life_sq'] > data['full_sq'], data['full_sq'])
    data['life_sq'] = data['life_sq'].mask(data['life_sq'].isna(), data['full_sq'])
    data['num_room'] = data['num_room'].mask(data['num_room'] > 9, 2)
    data['num_room'] = data['num_room'].mask(data['num_room'] < 1, data['num_room'].median())
    return data

def clean_level_2(data, train = True):
    data['floor'] = data['floor'].mask(data['floor'].isna(), data['max_floor'])
    data['max_floor'] = data['max_floor'].mask(data['max_floor'].isna(), data['floor'])
    data['floor'] = data['floor'].fillna(data['floor'].median())
    data['max_floor'] = data['max_floor'].fillna(data['max_floor'].median())
    if train:
        data = data.drop((data[data['price_doc'] / data['full_sq'] > 0.6 * 1e6]).index.tolist())
    return data

def add_dates(data):
    #Add year, month and week:
    data['sell_year'] = pd.DatetimeIndex(data['timestamp']).year.tolist()
    data['sell_month'] = pd.DatetimeIndex(data['timestamp']).month.tolist()
    data['sell_week'] = pd.DatetimeIndex(data['timestamp']).isocalendar().week.tolist()
    # Add month year count:
    month_year = (data.timestamp.dt.month + data.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    data['month_year_cnt'] = month_year.map(month_year_cnt_map)
    # Add week year count:
    week_year = (data.timestamp.dt.isocalendar().week + data.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    data['week_year_cnt'] = week_year.map(week_year_cnt_map)
    return data.drop('timestamp', axis=1)

def add_misc_features(data):
    data['frac_sq'] = data['life_sq'] / data['full_sq']
    data['frac_floor'] = data['floor'] / data['max_floor']
    data['mean_brent'] = data['sell_year'].map(dict(data.groupby('sell_year')['brent'].mean()))
    return data