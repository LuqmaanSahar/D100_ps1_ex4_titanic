import pandas as pd
import numpy as np

TRAIN_PATH = "titanic_package/data/train.csv"
TEST_PATH = "titanic_package/data/test.csv"

def data_load(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """
    Load both .csv datasets

    Returns:
    ----------
    train_df: (pd.DataFrame) training data
    test_df: (pd.(Dataframe) testing data

    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def data_missing(df):
    """
    Returns the missing values per column/feature in the dataframe

    Parameters:
    ----------
    df: (pd.DataFrame) dataset to summarise

    Returns:
    ----------
    df_missing: (ndarray) array of summary statistics per feature

    """
    total = df.isnull().sum()
    percent = (df.isnull().sum()/df.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    df_missing = np.transpose(tt)
    return df_missing


def data_frequent(df):
    """
    Returns a table summarising the most frequent data in each column/feature

    Parameters:
    ----------
    df: (pd.DataFrame) dataset to summarise


    Returns:
    ----------
    df_frequency: (ndarray) array of summary statistics per feature

    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]
            val = df[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    df_frequency = np.transpose(tt)
    return df_frequency


def data_unique(df):
    """
    Returns a table of the number of unique values per column/feature

    Parameters:
    ----------
    df: (pd.DataFrame) dataset to summarise

    
    Returns:
    ----------
    df_unique: (ndarray) count of unique values for each column/feature
    
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    df_unique = np.transpose(tt)
    return df_unique
