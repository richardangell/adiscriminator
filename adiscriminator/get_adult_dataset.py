import pandas as pd
import numpy as np
import os.path


def get_data(verbose = 0):
    
    file_path = 'adult.pkl'
    
    if os.path.exists(file_path):
        
        if verbose > 0:

            print('loading adult dataset from .pkl')

        data = pd.read_pickle(file_path)

    else:

        if verbose > 0:

            print('downloading adult dataset from uci ml repository')

        data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        
        data = pd.read_csv(data_url)
        
        data.columns = ['age',
                        'workclass',
                        'fnlwgt',
                        'education',
                        'education_num',
                        'marital_status',
                        'occupation',
                        'relationship',
                        'race',
                        'sex',
                        'capital_gain',
                        'capital_loss',
                        'hours_per_week',
                        'native_country',
                        'income']
        
        data['income'] = (data['income'] == ' <=50K') * 1

        if verbose > 0:

            print('saving adult dataset to .pkl')

        data.to_pickle(file_path)

    return(data)


def data_to_np(data):

    # initially only keep the numeric columns
    keep_cols = ['age',
                 #'workclass',
                 'fnlwgt',
                 #'education',
                 'education_num',
                 #'marital_status',
                 #'occupation',
                 #'relationship',
                 #'race',
                 #'sex',
                 'capital_gain',
                 'capital_loss',
                 'hours_per_week',
                 #'native_country'
                 ]

    X = np.array(data.loc[:, keep_cols])

    y = np.array(data.loc[:, 'income'])

    return(X, y)


if __name__ == '__main__':

    adult = get_data()

    print(adult.head())

    X, y = data_to_np(adult)


