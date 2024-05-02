import pandas as pd
# import os
import s3fs
import json
from scipy import sparse

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from dask import dataframe as dd
import ast
from numpy.linalg import norm
from src.GloVe.weights import *
import warnings
import os
import yaml

# Create filesystem object
s3_config = yaml.safe_load(open("S3_config.yml"))
ssp_cloud = True
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': s3_config['endpoint_url']},key = s3_config["key"], secret = s3_config["secret"], token = s3_config["token"])
bucket = "nfarhan/diffusion/mise_en_production"


BUCKET = "nfarhan/diffusion/mise_en_production"
FILE_KEY_S3 = "with_parliament/current_dataframes/df_BT.csv"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

# # open csv as dataframe
# with fs.open(FILE_PATH_S3, mode="r") as file_in:
#     df_BT = pd.read_csv(file_in, sep=",")

# df_BT = pd.read_csv(file_in, sep=",")

# # open json file
# with fs.open(BUCKET + "/" + 'with_parliament/vocabs/vocab_2016.json', mode="r") as file_in:
#     vocab_year = json.load(file_in)

# # open txt file
# with fs.open(BUCKET + "/" + 'with_parliament/sentence_embeddings/sentence_embeddings_2010.txt', mode="r") as file_in:
#     text_content = file_in.read()

# # open npz files (glove_coocurences)
# with fs.open(BUCKET + "/" + 'with_parliament/glove_cooccurences/glove_cooccurrence_2010.npz', mode="rb") as file_in:
#     cooccurrence = sparse.load_npz(file_in).toarray()

#print(df_BT.head())

fichier = bucket+"/with_parliament/sentence_embeddings/sentence_embeddings_" + str(2013) + ".txt"

with fs.open(fichier, mode='r') as f:
    text_content = f.read()

local_file_path = 'local_file.txt'  # Specify the local file path where you want to save the content
with open(local_file_path, mode="w") as file_out:
    file_out.write(text_content)

word2vec_glove_file_sentences = get_tmpfile("format_word2vec.text")
glove2word2vec(local_file_path, word2vec_glove_file_sentences)
with open(word2vec_glove_file_sentences, "r") as file:
    data = file.readlines()
    data[0] = str(len(data) - 1) + " 50\n"
with open(word2vec_glove_file_sentences, "w") as file:
    file.writelines(data)
a = KeyedVectors.load_word2vec_format(word2vec_glove_file_sentences)

# Step 3: Delete the local file
os.remove(local_file_path)

