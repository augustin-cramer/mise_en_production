from GloVe.weights import *
import warnings
import os

warnings.filterwarnings("ignore")
from Axes.projection_functions import *
from Axes.axes_definition import *

os.chdir("../")
print(os.getcwd())

ssp_cloud=False
fs=None
bucket=None

# PART I: Generation of word2vec models for sentence and word embeddings

# Generate word2vec models for sentences from text files
models_s = []  # List to store sentence models
for i in range(14):  # Assuming 14 years of data, from 2010 to 2023
    # Construct the file path for sentence embeddings
    file_path = f"/without_parliament/sentence_embeddings/sentence_embeddings_201{i}.txt"
    # Load the text data and convert it into a word2vec model for sentences
    models_s.append(txt_to_model_sentences(file_path, ssp_cloud, fs, bucket))

# Generate word2vec models for words from text files
models_w = []  # List to store word models
for i in range(14):  # Loop through the same range for word embeddings
    # Construct the file path for word embeddings
    file_path = f"/without_parliament/embeddings/embeddings_201{i}_WP.txt"
    # Load the text data and convert it into a word2vec model for words
    models_w.append(txt_to_model_words(file_path, ssp_cloud, fs, bucket))

# Change the directory back to 'src' to continue with script execution
os.chdir(r"src")
