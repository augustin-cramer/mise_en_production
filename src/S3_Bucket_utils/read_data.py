import pandas as pd
import json
from scipy import sparse


def read_csv_bucket(fs, FILE_PATH_S3):
    """
    Reads a CSV file from an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        FILE_PATH_S3 (str): Path to the CSV file in the object storage system.

    Returns:
        DataFrame: A pandas DataFrame containing the CSV data.
    """
    with fs.open(FILE_PATH_S3, mode="r") as file_in:
        df = pd.read_csv(file_in, sep=",")
    return df


def read_json_bucket(fs, FILE_PATH_S3):
    """
    Reads a JSON file from an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        FILE_PATH_S3 (str): Path to the JSON file in the object storage system.

    Returns:
        dict: A dictionary containing the JSON content.
    """
    with fs.open(FILE_PATH_S3, mode="r") as file_in:
        json_content = json.load(file_in)
    return json_content


def read_txt_bucket(fs, FILE_PATH_S3):
    """
    Reads a text file from an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        FILE_PATH_S3 (str): Path to the text file in the object storage system.

    Returns:
        str: The content of the text file.
    """
    with fs.open(FILE_PATH_S3, mode="r") as file_in:
        text_content = file_in.read()
    return text_content


def read_npz_bucket(fs, FILE_PATH_S3):
    """
    Reads a NumPy's compressed sparse matrix file (NPZ) from an object storage system.

    Parameters:
        fs (object): File system object for accessing storage.
        FILE_PATH_S3 (str): Path to the NPZ file in the object storage system.

    Returns:
        array: The content of the NPZ file as a NumPy array.
    """
    with fs.open(FILE_PATH_S3, mode="rb") as file_in:
        npz_content = sparse.load_npz(file_in).toarray()
    return npz_content
