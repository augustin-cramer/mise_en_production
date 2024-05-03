import pandas as pd
import json
from scipy import sparse
import streamlit as st
import os
from gensim.scripts.glove2word2vec import glove2word2vec


@st.cache_data
def read_csv_bucket(_connection, file_path_s3, **kwargs):
    """
    Reads a CSV file from an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        file_path_s3 (str): Path to the CSV file in the object storage system.

    Returns:
        DataFrame: A pandas DataFrame containing the CSV data.
    """
    with _connection["fs"].open(
        _connection["bucket"] + file_path_s3, mode="r"
    ) as file_in:
        df = pd.read_csv(file_in, sep=",", **kwargs)
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
    with _connection["fs"].open(
        _connection["bucket"] + file_path_s3, mode="r"
    ) as file_in:
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
    with _connection["fs"].open(
        _connection["bucket"] + file_path_s3, mode="r"
    ) as file_in:
        text_content = file_in.readlines()
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
    with _connection["fs"].open(
        _connection["bucket"] + file_path_s3, mode="rb"
    ) as file_in:
        npz_content = sparse.load_npz(file_in).toarray()
    return npz_content


def write_csv_bucket(_connection, df, file_path_s3):
    """
    Writes a DataFrame to a CSV file in an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        df (DataFrame): The DataFrame to be written to a CSV file.
        file_path_s3 (str): Path to the CSV file in the object storage system.
    """
    _connection["fs"].makedirs(
        os.path.dirname(_connection["bucket"] + file_path_s3), exist_ok=True
    )
    with _connection["fs"].open(
        _connection["bucket"] + file_path_s3, "w"
    ) as file_out:
        df.to_csv(file_out, index=False)


def write_txt_bucket(_connection, text, file_path_s3):
    """
    Writes a text to a text file in an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        text (str): The text to be written to a text file.
        file_path_s3 (str): Path to the text file in the object storage system.
    """
    _connection["fs"].makedirs(
        os.path.dirname(_connection["bucket"] + file_path_s3), exist_ok=True
    )
    with _connection["fs"].open(
        _connection["bucket"] + file_path_s3, "w"
    ) as file_out:
        file_out.writelines(text)


class DataLoader:
    def __init__(self, connection):
        self.from_s3 = False
        if connection is not None:
            self.from_s3 = True
        self.connection = connection

    def read_csv(self, file_path, **kwargs):
        if self.from_s3:
            return read_csv_bucket(self.connection, file_path, **kwargs)
        else:
            return pd.read_csv("data/" + file_path, **kwargs)

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
                return file.readlines()

    def read_npz(self, file_path):
        if self.from_s3:
            return read_npz_bucket(self.connection, file_path)
        else:
            with open("data/" + file_path, "rb") as file:
                return sparse.load_npz(file).toarray()

    def exists(self, file_path):
        if self.from_s3:
            full_file_path = self.connection["bucket"] + file_path
            return full_file_path in self.connection["fs"].ls(
                self.connection["bucket"]
            )
        else:
            return os.path.exists("data/" + file_path)

    def write_csv(self, df, file_path):
        if self.from_s3:
            write_csv_bucket(self.connection, df, file_path)
        else:
            os.makedirs(os.path.dirname("data/" + file_path), exist_ok=True)
            df.to_csv("data/" + file_path, index=False)

    def write_txt(self, text, file_path):
        if self.from_s3:
            write_txt_bucket(self.connection, text, file_path)
        else:
            os.makedirs(os.path.dirname("data/" + file_path), exist_ok=True)
            with open("data/" + file_path, "w") as file:
                file.writelines(text)

    def glove2word2vec(self, file_path, word2vec_glove_file_sentences):
        if self.from_s3:
            os.makedirs(os.path.dirname("data/" + file_path), exist_ok=True)
            text = self.read_txt(file_path)
            with open("data/" + file_path, "w") as file:
                file.writelines(text)
            glove2word2vec(
                "data/" + file_path,
                word2vec_glove_file_sentences,
            )
        else:
            glove2word2vec("data/" + file_path, word2vec_glove_file_sentences)
