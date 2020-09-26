#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from env import host, user, password
import os
from pydataset import data


# In[14]:
#### lets acquire mall customers data

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    


# In[15]:


def new_titanic_data():
    '''
    This function reads the titanic data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    sql_query = 'SELECT * FROM passengers'
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    df.to_csv('titanic_df.csv')
    return df


def get_titanic_data(cached=False):
    '''
    This function reads in titanic data from Codeup database if cached == False 
    or if cached == True reads in titanic df from a csv file, returns df
    '''
    if cached or os.path.isfile('titanic_df.csv') == False:
        df = new_titanic_data()
    else:
        df = pd.read_csv('titanic_df.csv', index_col=0)
    return df


def new_telco_data():
    '''This function reads the telco  data from the Codeup db into a df,
    write it to a csv file, and returns the df.'''
    query = '''select c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents, c.tenure, c.phone_service,
        c.multiple_lines, c.internet_service_type_id, c.online_security, c.online_backup,c.device_protection, c.tech_support,
        c.streaming_tv, c.streaming_movies, c.contract_type_id, c.paperless_billing, c.payment_type_id, c.monthly_charges, c.total_charges,c.churn, ct.contract_type, ist.internet_service_type, pt.payment_type
        from customers c
        join contract_types ct on ct.contract_type_id = c.contract_type_id
        join internet_service_types ist on ist.internet_service_type_id = c.internet_service_type_id
        join payment_types pt on pt.payment_type_id = c.payment_type_id'''
    telco = pd.read_sql(query, get_connection('telco_churn'))
    telco.to_csv('telco.csv')
    return telco




def get_telco_data(cached=False):
    '''
    This function reads in titanic data from Codeup database if cached == False 
    or if cached == True reads in titanic df from a csv file, returns df
    '''
    if cached or os.path.isfile('telco.csv') == False:
        telco = new_telco_data()
    else:
        telco = pd.read_csv('telco.csv', index_col=0)
    return telco


def new_mall_data():

    ''' This function reads the mall data from codeup database into a df,
     write it to a cv file, and returns the df '''
     
    sql_query = 'Select * FROM customers'
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    df.to_csv('mall_customers_df.csv')
    return df


def get_mall_data(cached=False):
    '''
    This function reads in mall customers data from Codeup database if cached == False 
    or if cached == True reads in mall customers  df from a csv file, returns df
    '''
    if cached or os.path.isfile('mall_customers_df.csv') == False:
        df = new_mall_data()
    else:
        df = pd.read_csv('mall_customers_df.csv', index_col=0)
    return df


# In[16]:




# In[17]:


def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df,
    writes it to a csv file, and returns the df.
    '''
    sql_query = """
                SELECT species_id,
                species_name,
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
                FROM measurements
                JOIN species
                USING(species_id)
                """
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    df.to_csv('iris_df.csv')
    return df


# In[18]:


def get_iris_data(cached=False):
    '''
    This function reads in iris data from Codeup database if cached == False
    or if cached == True reads in iris df from a csv file, returns df
    '''
    if cached or os.path.isfile('iris_df.csv') == False:
        df = new_iris_data()
    else:
        df = pd.read_csv('iris_df.csv', index_col=0)
    return df

