#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import acquire
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy import stats


# In[2]:


def prep_iris():
    df = acquire.get_iris_data()
    cols_to_drop = ['species_id']
    df = df.drop(columns = cols_to_drop)
    df = df.rename(columns = {"species_name":"species"})
    df_dummies = df_dummies = pd.get_dummies(df[['species']], drop_first = True)
    df = pd.concat([df, df_dummies], axis = 1)
    return df


# In[3]:


def split_data(df):
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=322, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size= 0.2,
                                       random_state = 322, stratify = train_validate.survived)
    return train, test, validate


def prep_titanic():
    titanic = acquire.get_titanic_data()
    titanic = titanic[~ titanic.embarked.isnull()]
    titanic = titanic[~ titanic.embark_town.isnull()]
    
    df_dummies = pd.get_dummies(titanic[['embark_town']], drop_first = True)
    
    df_dum = pd.get_dummies(titanic[['sex']], drop_first = True)
    
    titanic = pd.concat([titanic, df_dummies, df_dum], axis = 1)
    
    
    cols_to_drop = ['passenger_id','pclass', 'embarked', 'deck', 'sex']
    titanic = titanic.drop(columns = cols_to_drop)
    
    train, test, validate = split_data(titanic)
    return train, test, validate


# In[1]:


def impute(train, test, validate, my_strategy, column_list):
    
    imputer = SimpleImputer(strategy = my_strategy)
    
    imputer = imputer.fit(train[[column_list]])
    
    train[column_list] = imputer.transform(train[[column_list]])
    
#     imputer = imputer.fit(test[[column_list]])
    
    test[column_list] = imputer.transform(test[[column_list]]) 
    
#     imputer = imputer.fit(validate[[column_list]])
    
    validate[column_list] = imputer.transform(validate[[column_list]])
    
    return train, test, validate


# In[ ]:
def prep_mall_data(df):
    '''
    Takes the acquired mall data, does data preparation, and return train, test, 
    and validate data splits.
    '''
    df = df.drop(columns = 'Unnamed: 0')
    df['is_female'] = (df.gender =='Female').astype('int')
    df = df.drop(columns = ['gender', 'customer_id'])
    train_validate, test = train_test_split(df, test_size =0.15, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = 0.15, random_state = 123)
    return train, test, validate

