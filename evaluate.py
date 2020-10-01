import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pydataset import data

from sklearn.metrics import mean_squared_error
from math import sqrt

# Linear Model
from statsmodels.formula.api import ols



def plot_residuals(actual, predicted):

    '''takes in actual values and predicted values and retuns an azes with
    actual value plotted agains the residuals'''
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()


def regression_errors(actual, predicted):

    '''takes in actual value and predicted values and returns
    sse, mse, rmse, ess, and tss'''
    df = pd.DataFrame()
    df['actual'] = actual
    df['predicted'] = predicted
    
    
    df['residuals'] = df.predicted - df.actual
    sse = (df.residuals ** 2).sum()
    mse = mean_squared_error(df.actual, df.predicted)
    rmse = sqrt(mse)
    ess = sum((df.predicted - df.actual.mean()) ** 2)
    tss = sse + ess
    
    return(sse, mse, rmse, ess, tss)
                 
                



def baseline_mean_errors(actual):
   

    '''takes in an actual value(dependent variable) and returns sse
    mse and rmse'''
    
    df = pd.DataFrame()
    df['actual'] = actual
    df['baseline'] = df.actual.mean()
    df['baseline_residual'] = df.baseline - df.actual
    sse = (df.baseline_residual ** 2).sum()
    mse = mean_squared_error(df.actual, df.baseline)
    rmse = sqrt(mse)
    
    return (sse, mse, rmse)



def better_than_baseline(actual, predicted):
    
    '''function that takes the actual value and the predicted value
    and returns if the model beats the baseline model based on the
    calculated sse '''
    df = pd.DataFrame()
    df['actual'] = actual
    df['baseline'] = df.actual.mean()
    df['baseline_residual'] = df.baseline - df.actual
    df['predicted'] = predicted
    
    baseline_sse = (df.baseline_residual ** 2).sum()
    
    
    df['model_residuals'] = df.predicted - df.actual
    model_sse = (df.model_residuals ** 2).sum()
    
    if model_sse < baseline_sse:
        print('model_sse:' , model_sse)
        print('baseline_sse:', baseline_sse)
        print("Our model beats the baseline")
        
    else:
        print('model_sse:' , model_sse)
        print('baseline_sse:', baseline_sse)
        print("Our baseline is better than the model.")
    




def model_significance(ols_model):

    '''takes in an ols model created and fitted and returns r2 and p value
    with a statement indicateing if the created model is better than the baseline based on 
    the r2 and p value'''

    alpha = 0.05
    
    r2 = ols_model.rsquared
    
    f_pval = ols_model.f_pvalue
    
    if f_pval < alpha:
        print('r2::', r2)
        print('p_val:', f_pval)
        print('Our model is sgnificantly better than baseline based on above results ')
    else:
        print('r2::', r2)
        print('p_val:', f_pval)
        print('Our model is not  better than baseline based on above results ')