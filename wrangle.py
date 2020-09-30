


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.model_selection import train_test_split




def wrangle_telco(df):
    df = df[df.contract_type == 'Two year']
    df = df[['customer_id', 'monthly_charges', 'tenure', 'total_charges']]
    df.total_charges = pd.to_numeric(df.total_charges, errors = 'coerce')
    df = df.dropna()
    return df
def split_data(df):
    train_validate, test = train_test_split(df, test_size = .15, random_state = 123)

    train, validate = train_test_split(train_validate, test_size = 0.15, random_state = 123)

    return train, test, validate

def scale_data(train, test, validate, scaler, cols_to_scale):
    '''Intakes train, test, validate dataset, scaling methond, and the list of columns to scale
    and returns train, test, validate dataset with scaled data added to new columns'''

    
    new_cols = [col + "_scaled" for col in cols_to_scale]
    
    
    scaler = scaler.fit(train[cols_to_scale])
    
    train = pd.concat([train, pd.DataFrame(scaler.transform(train[cols_to_scale]), 
                                           index = train.index, columns = new_cols)], axis = 1)
    
    
    test =  pd.concat([test, pd.DataFrame(scaler.transform(test[cols_to_scale]), 
                                           index = test.index, columns = new_cols)], axis = 1)
    
    
    validate =  pd.concat([validate, pd.DataFrame(scaler.transform(validate[cols_to_scale]), 
                                           index = validate.index, columns = new_cols)], axis = 1)
                                        
                      
    return train, test, validate     







def wrangle_grades():
    grades = pd.read_csv("student_grades.csv")
    grades.drop(columns='student_id', inplace=True)
    grades.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df = grades.dropna().astype('int')
    return df