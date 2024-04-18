import pandas as pd
# import os
import s3fs

# Create filesystem object
S3_ENDPOINT_URL = "https://minio.lab.sspcloud.fr"
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


BUCKET = "nfarhan/diffusion/mise_en_production"
FILE_KEY_S3 = "with_parliament/current_dataframes/df_BT.csv"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

with fs.open(FILE_PATH_S3, mode="rb") as file_in:
    df_BT = pd.read_csv(file_in, sep=",")

print(df_BT.head())
