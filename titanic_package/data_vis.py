import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

def agg_data(tr_df, te_df):

    """
    Returns a new data frame, which combines both the training and the test data. Also creates a new column called set. 

    Parameters:
    ----------
    tr_df: (pd.DataFrame) the training dataframe 
    tr_df: (pd.DataFrame) the test dataframe  
    
    Returns:
    ----------
    all_df: (pd.DataFrame) the combined dataframe
    
    """

    all_df = pd.concat([tr_df, te_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df

def calc_fam_size(df): 

    """
    Creates a new column 'Family Size', which calculates the family size based on the SibSp and Parch columns 

    Parameters:
    ----------
    df: (pd.DataFrame) dataset to summarise
    
    """

    df["Family Size"] = df["SibSp"] + df["Parch"] + 1

def age_interval(df): 

    """
    Creates a new column 'Age_Interval', which is a number depending on the age interval of the row

    Parameters:
    ----------
    df: (pd.DataFrame) dataset to summarise
    
    """
    
    df["Age Interval"] = 0.0
    df.loc[ df['Age'] <= 16, 'Age Interval']  = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[ df['Age'] > 64, 'Age Interval'] = 4

def fare_interval(df):
    
    """
    Creates a new column 'Fare Interval', which is a number depending on the fare interval the row falls in 

    Parameters:
    ----------
    df: (pd.DataFrame) dataset to summarise

    """

    df['Fare Interval'] = 0.0
    df.loc[ df['Fare'] <= 7.91, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare Interval']   = 2
    df.loc[ df['Fare'] > 31, 'Fare Interval'] = 3

def sex_pclass(df): 
      
    """
    Creates a new column 'Sex_Pclass', which combines the 'Sex' and the 'PClass' columns

    Parameters:
    ----------
    df: (pd.DataFrame) dataset to summarise
    
    """
    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)