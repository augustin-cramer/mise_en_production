"""This script loads the embeddings from text format 
into word2vec format, which is much more manipulable."""

from ..Axes.projection_functions import txt_to_model_words, txt_to_model_sentences

import warnings

warnings.filterwarnings("ignore")

# PART I: Generation of word2vec models for sentence and word embeddings

def instatiate_models_s():
    # Generate word2vec models for sentences from text files
    models_s = []  # List to store sentence models
    for i in range(14):  # Assuming 14 years of data, from 2010 to 2023
        # Construct the file path for sentence embeddings
        file_path = f"data/without parliament/sentence_embeddings/sentence_embeddings_201{i}.txt"
        # Load the text data and convert it into a word2vec model for sentences
        models_s.append(txt_to_model_sentences(file_path))

def instatiate_models_w():
    # Generate word2vec models for words from text files
    models_w = []  # List to store word models
    for i in range(14):  # Loop through the same range for word embeddings
        # Construct the file path for word embeddings
        file_path = f"data/without parliament/embeddings/embeddings_201{i}_WP.txt"
        # Load the text data and convert it into a word2vec model for words
        models_w.append(txt_to_model_words(file_path))
