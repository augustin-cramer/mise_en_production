import pandas as pd
import json
from scipy import sparse
import streamlit as st

@st.cache_data
def read_csv_bucket(_connection, file_path_s3):
    """
    Reads a CSV file from an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        file_path_s3 (str): Path to the CSV file in the object storage system.

    Returns:
        DataFrame: A pandas DataFrame containing the CSV data.
    """
    with _connection["fs"].open(_connection["bucket"] + file_path_s3, mode="r") as file_in:
        df = pd.read_csv(file_in, sep=",")
    return df

@st.cache_data
def read_json_bucket(_connection, file_path_s3):
    """
    Reads a JSON file from an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        file_path_s3 (str): Path to the JSON file in the object storage system.

    Returns:
        dict: A dictionary containing the JSON content.
    """
    with _connection["fs"].open(_connection["bucket"] + file_path_s3, mode="r") as file_in:
        json_content = json.load(file_in)
    return json_content

@st.cache_data
def read_txt_bucket(_connection, file_path_s3):
    """
    Reads a text file from an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        file_path_s3 (str): Path to the text file in the object storage system.

    Returns:
        str: The content of the text file.
    """
    with _connection["fs"].open(_connection["bucket"] + file_path_s3, mode="r") as file_in:
        text_content = file_in.read()
    return text_content

@st.cache_data
def read_npz_bucket(_connection, file_path_s3):
    """
    Reads a NumPy's compressed sparse matrix file (NPZ) from an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        FILE_PATH_S3 (str): Path to the NPZ file in the object storage system.

    Returns:
        array: The content of the NPZ file as a NumPy array.
    """
    with _connection["fs"].open(_connection["bucket"] + file_path_s3, mode="rb") as file_in:
        npz_content = sparse.load_npz(file_in).toarray()
    return npz_content

class DataLoader():
    def __init__(self, connection):
        if connection is not None:
            self.from_s3 = True
        self.connection = connection
    
    def read_csv(self, file_path):
        if self.from_s3:
            return read_csv_bucket(self.connection, file_path)
        else:
            return pd.read_csv("data/" + file_path)
        
    def read_json(self, file_path):
        if self.from_s3:
            return read_json_bucket(self.connection, file_path)
        else:
            with open("data/" + file_path) as file:
                return json.load(file)
    
    def read_txt(self, file_path):
        if self.from_s3:
            return read_txt_bucket(self.connection, file_path)
        else:
            with open("data/" + file_path) as file:
                return file.read()
        
    def read_npz(self, file_path):
        if self.from_s3:
            return read_npz_bucket(self.connection, file_path)
        else:
            with open("data/" + file_path, "rb") as file:
                return sparse.load_npz(file).toarray()