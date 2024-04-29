import pandas as pd
# import os
import s3fs
import json
from scipy import sparse

# Create filesystem object
S3_ENDPOINT_URL = "https://minio.lab.sspcloud.fr"
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


BUCKET = "nfarhan/diffusion/mise_en_production"
FILE_KEY_S3 = "with_parliament/current_dataframes/df_BT.csv"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

# open csv as dataframe
with fs.open(FILE_PATH_S3, mode="r") as file_in:
    df_BT = pd.read_csv(file_in, sep=",")

# open json file
with fs.open(BUCKET + "/" + 'with_parliament/vocabs/vocab_2016.json', mode="r") as file_in:
    vocab_year = json.load(file_in)

# open txt file
with fs.open(BUCKET + "/" + 'with_parliament/sentence_embeddings/sentence_embeddings_2010.txt', mode="r") as file_in:
    text_content = file_in.read()

# open npz files (glove_coocurences)
with fs.open(BUCKET + "/" + 'with_parliament/glove_cooccurences/glove_cooccurrence_2010.npz', mode="rb") as file_in:
    cooccurrence = sparse.load_npz(file_in).toarray()

print(df_BT.head())
