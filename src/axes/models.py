"""This script loads the embeddings from text format 
into word2vec format, which is much more manipulable."""

from ..axes.projection_functions import txt_to_model_words

import warnings

warnings.filterwarnings("ignore")


def instatiate_models_w(data_loader):
    # Generate word2vec models for words from text files
    models_w = []  # List to store word models
    for i in range(14):  # Loop through the same range for word embeddings
        # Construct the file path for word embeddings
        file_path = f"without_parliament/embeddings/embeddings_201{i}_WP.txt"
        # Load the text data and convert it into a word2vec model for words
        models_w.append(txt_to_model_words(data_loader, file_path))
    return models_w
