import os
import pandas as pd
import numpy as np
import matlab.engine

def listdir_clean(fpath):
    """
    Returns list of files in a directory, eliminating any hidden files (e.g. 
    the pesky .DS_Store on mac)

    Args:
        fpath (str): path of folder

    Returns:
        list: list of folder contents
    """
    return [f for f in os.listdir(fpath) if not f.startswith('.')]


def load_table_dt(fpath, dt_col_name, usecols=None):
    """
    Returns a pandas DataFrame containing the data from csv with path fpath. 
    Specified column is converted to pandas datetime.

    Args:
        fpath (str): Filepath of csv file to be read
        dt_col_name (str): Name of column in csv file to be read in as a
            datetime
        uescols (bool): Determine which columns to read in from the csv

    Returns:
        DataFrame: contents of csv file
    """
    # Make sure that all cols are read in as strings for consistency
    dtype = {usecols[i]: str for i in range(len(usecols))}
    data = pd.read_csv(fpath, usecols=usecols, dtype=dtype)
    for col in usecols:
        data[col] = data[col].str.strip()
        if col == dt_col_name:
            data[col] = pd.to_datetime(data[col])
        else:
            data[col] = data[col].astype(np.float64)
    return data.dropna()


def inrange(query, min, max):
    """
    Shortcut function to see if a query point is in a range given by max and 
    min. Used for timeseries values in preprocessing

    Args:
        query (int, float, datetime): point to query
        min (int, float, datetime): min point of range of comparison
        max (int, float, datetime): max point of range of comparison

    Returns:
        bool: Indicates if query is in the specified range
    """
    return query >= min and query <= max


def timedelta_to_seconds(tdelta):
    """Convert TimeDelta datatype to float seconds

    Args:
        tdelta (TimeDelta): TimeDelta value

    Returns:
        float: seconds value
    """
    return tdelta.astype(int) / 1e9


def matlab_to_numpy(matlab_arr):
    """Convert Matlab array to numpy array

    Args:
        matlab_arr (matlab.double): python Matlab array datatype

    Returns:
        np.ndarray: ndarray equivalent of matlab double array
    """
    return np.array(matlab_arr)


def numpy_to_matlab(numpy_arr):
    """Converts numpy array to corresponding matlab array

    Args:
        numpy_arr (np.ndarray): numpy array

    Returns:
        matlab.double: matlab equivalent of numpy array
    """
    return matlab.double(numpy_arr.tolist())