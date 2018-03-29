import pandas as pd
import os.path


def get_data():
    
    #file_path
    
    #if os.path.exists(file_path)
    
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
    
    return(data)


def save_data():


a.to_pickle('adult.pkl')
b = pd.read_pickle('adult.pkl')

if __name__ == '__main__':

    get_data()

