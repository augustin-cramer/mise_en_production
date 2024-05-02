"""Contains the functions to compute the weighting of embeddings 
inside a document in order to get the document general embedding."""

import ast
import gc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


def standard_opening(fichier, agenda: bool, ssp_cloud=False, fs=None, bucket=None):
    """Standardizes the opening of dataframes that we use in the following of the study

    Parameters:
    -----------
    fichier : the dataframe to open
    agenda : says if the dataframe contains an agenda column (used for paliament)
    """
    if ssp_cloud:
        with fs.open(bucket+fichier, mode="r") as file_in:
            df = pd.read_csv(file_in, index_col=[0])
    else:
        df = pd.read_csv("data"+fichier, index_col=[0])
    df["text"] = df["text"].map(ast.literal_eval)
    df["keywords"] = df["keywords"].map(ast.literal_eval)
    if agenda:
        df["agenda"] = df["agenda"].map(ast.literal_eval)
    return df


def get_weights_word2vec(words, a=1e-3):
    """Determines the weights that we will attribute to each
    word in our model to compute sentence embeddings.
    The weights formula comes from a MLE

    Parameters:
    -----------
    words : the set of all words used in the corpuses, with repetitions
    a : weight parameter, empirically found optimal at 1e-3
    """
    vectorizer = CountVectorizer(decode_error="ignore")
    counts = vectorizer.fit_transform(words)
    total_freq = np.sum(counts, axis=0).T
    N = np.sum(total_freq)
    weighted_freq = a / (a + total_freq / N)
    gc.collect()
    return dict(zip(vectorizer.get_feature_names_out(), weighted_freq))


def get_sentence_embeddings(phrase, weights, model):
    """Uses the weights of each word to form a sentence embedding
    Parameters:
    -----------
    phrase : the sentence
    weights : the dictionary of weights
    model :  the word2vec model containing the embeddings of all the words
    """
    sentence_embedding = 0
    for word in phrase:
        try:
            sentence_embedding = sentence_embedding + (
                weights[word] * model[word]
            )
        except:
            sentence_embedding += 0
    try:
        return [
            embed / (len(phrase)) for embed in sentence_embedding.tolist()[0]
        ]
    except:
        return None


def phrase(list):
    """Creates a sentence linked with '_' from a list of words.
    Used for sentence recognition in word2vec models

    Parameters:
    -----------
    list : list of words forming the sentence
    """
    phr = ""
    for string in list:
        phr += string + "_"
    return phr


def barycentre(list, model):
    """Computes the embdeddings barycenter from a list of words
    or sentences

    Parameters:
    -----------
    list : list of words or sentences
    model : the word2vec model containing the embeddings
    """
    bar = np.zeros(50, dtype=object)
    for word in list:
        bar = bar + model[word]
    return bar / (len(list))


def compute_pc(X, npc=1):
    """
    Compute the principal components of X. DO NOT MAKE THE DATA
    ZERO MEAN!

    Parameters:
    -----------
    X: X[i,:] is a data point
    npc: number of principal components to remove
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    print("Computing principal components...")
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components

    Parameters:
    -----------
    X: X[i,:] is a data point
    npc: number of principal components to remove
    """
    print("Removing principal component...")
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX
