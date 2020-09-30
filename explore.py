import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import acquire
import wrangle
import sklearn.preprocessing
import math








def plot_variables_pairs(df):
    g = sns.PairGrid(df)
    g.map_diag(plt.hist)
    g.map_offdiag(sns.regplot)



def months_to_years(df):
    return df.assign(tenure_years = (df.tenure/12).apply(math.floor))



def plot_categorical_and_continuous(df, continuous_variables, categorical_variables):
    for col in continuous_variables:
       
        for c in categorical_variables:
            
            sns.barplot(data = df, y = col, x = c)
           
            plt.show()
            
            sns.violinplot(data = df, x = c, y = col)
           
            plt.show()
            
            
            sns.swarmplot(data = df, x = c, y = col)
            
            plt.show()
            
            sns.boxplot(data = df, x = c, y = col)
            
            plt.show()







