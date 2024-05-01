"""This script loads the embeddings from text format 
into word2vec format, which is much more manipulable."""

from ..Axes.projection_functions import *
from ..Load_Data.load_data import *

import warnings

warnings.filterwarnings("ignore")
import s3fs
import yaml
import os

project_root = os.path.dirname(os.path.abspath(os.getcwd()))
yaml_file_path = os.path.join(project_root, "mise_en_production", "S3_config.yml")
s3_config = yaml.safe_load(open(yaml_file_path))
#s3_config = yaml.safe_load(open("S3_config.yml"))
ssp_cloud = True
fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": s3_config["endpoint_url"]},
    key=s3_config["key"],
    secret=s3_config["secret"],
    token=s3_config["token"],
)
bucket = "nfarhan/diffusion/mise_en_production"


# PART I: Generation of word2vec models for sentence and word embeddings

# Generate word2vec models for sentences from text files
models_s = []  # List to store sentence models
for i in range(14):  # Assuming 14 years of data, from 2010 to 2023
    # Construct the file path for sentence embeddings
    file_path = f"/without_parliament/sentence_embeddings/sentence_embeddings_201{i}.txt"
    # Load the text data and convert it into a word2vec model for sentences
    models_s.append(load_txt_model_sentence(file_path, ssp_cloud, fs, bucket))

# Generate word2vec models for words from text files
models_w = []  # List to store word models
for i in range(14):  # Loop through the same range for word embeddings
    # Construct the file path for word embeddings
    file_path = f"/without_parliament/embeddings/embeddings_201{i}_WP.txt"
    # Load the text data and convert it into a word2vec model for words
    models_w.append(load_txt_model_sentence(file_path, ssp_cloud, fs, bucket))

