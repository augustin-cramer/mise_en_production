from ..S3_Bucket_utils.read_data import *
from ..GloVe.weights import standard_opening
import pandas as pd
import json
from scipy import sparse
from ..Axes.projection_functions import *


def load_csv(filepath, ssp_cloud=False, fs=None, bucket=None):
    if ssp_cloud:
        df = read_csv_bucket(fs, bucket+filepath)
    else:
        df = pd.read_csv("data"+filepath)
    return df


def load_csv_index_col(filepath, ssp_cloud=False, fs=None, bucket=None):
    if ssp_cloud:
        with fs.open(bucket+filepath, mode="r") as file_in:
            df = pd.read_csv(file_in, index_col=[0])
    else:
        df = pd.read_csv("data"+filepath, index_col=[0])
    return df


def load_json(filepath, ssp_cloud=False, fs=None, bucket=None):
    if ssp_cloud:
        json_content = read_json_bucket(fs, bucket+filepath)
    else:
        with open("data"+filepath) as f:
            json_content = json.load(f)
    return json_content


def load_df_and_df_BT(with_parliament, ssp_cloud=False, fs=None, bucket=None):
    if with_parliament:
        filepath_df = "/with_parliament/current_dataframes/df.csv"
        filepath_df_BT = "/with_parliament/current_dataframes/df_BT.csv"
    if not with_parliament:
        filepath_df = "/without_parliament/current_dataframes/df.csv"
        filepath_df_BT = "/without_parliament/current_dataframes/df.csv"
    df = load_csv(filepath_df, ssp_cloud, fs, bucket)
    df_BT = load_csv(filepath_df_BT, ssp_cloud, fs, bucket)
    return df, df_BT


def load_json_vocab(with_parliament, year, ssp_cloud=False, fs=None, bucket=None):
    if with_parliament:
        filepath = "/with_parliament/vocabs/vocab_" + str(year) + ".json"
    if not with_parliament:
        filepath = "/without_parliament/vocabs/vocab_" + str(year) + "_WP.json"
    vocab = load_json(filepath, ssp_cloud, fs, bucket)
    return vocab


def load_current_year_data(with_parliament, year, ssp_cloud=False, fs=None, bucket=None):
    if with_parliament:
        df = standard_opening(
            f"/with_parliament/FinalDataframes/FilteredFinalDataFrame_201{i}.csv",
            True, ssp_cloud, fs, bucket
        ).reset_index()
    if not with_parliament:
        df = standard_opening(
            f"/without_parliament/FinalDataframes/FilteredFinalDataFrame_201{i}_WP.csv",
            True,  ssp_cloud, fs, bucket
        ).reset_index()
        df["party"], df["Speaker"] = 0, 0
    return df


def load_words_year(with_parliament, year, ssp_cloud=False, fs=None, bucket=None):
    if with_parliament:
        filepath = f"/with_parliament/words/Finalwords_{year}.json"
    if not with_parliament:
        filepath = f"/without_parliament/words/Finalwords_{year}.json"
    words_year = load_json(filepath, ssp_cloud, fs, bucket)
    return words_year


def load_vocab_year(with_parliament, year, ssp_cloud=False, fs=None, bucket=None):
    if with_parliament:
        filepath = f"/with_parliament/vocabs/vocab_{year}.json"
    if not with_parliament:
        filepath = f"/without_parliament/vocabs/vocab_{year}_WP.json"
    vocab_year = load_json(filepath, ssp_cloud, fs, bucket)
    return vocab_year


def load_finalwords(with_parliament, year, ssp_cloud=False, fs=None, bucket=None):
    if with_parliament:
        filepath = f"/with_parliament/words/Finalwords_{year}.json"
    if not with_parliament:
        filepath = f"/without_parliament/words/Finalwords_{year}_WP.json"
    words = load_json(filepath, ssp_cloud, fs, bucket)
    return words


def load_txt_model_sentence(filepath, ssp_cloud=False, fs=None, bucket=None):
    if ssp_cloud:
        with fs.open(bucket+filepath, mode='r') as f:
            text_content = f.read()

        local_file_path = 'local_file.txt' 
        with open(local_file_path, mode="w") as file_out:
            file_out.write(text_content)
        model_sentences = txt_to_model_sentences(
            local_file_path
        )
        os.remove(local_file_path)
        
    if not ssp_cloud:
        model_sentences = txt_to_model_sentences(
            "data" + filepath
        )
    return model_sentences


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
