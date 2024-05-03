"""Contains the functions to compute the weighting of embeddings 
inside a document in order to get the document general embedding."""

import ast
import gc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


def standard_opening(data_loader, fichier, agenda: bool):
    """Standardizes the opening of dataframes that we use in
    the following of the study

    Parameters:
    -----------
    fichier : the dataframe to open
    agenda : says if the dataframe contains an agenda column (used for paliament)
    """
    df = data_loader.read_csv(fichier, index_col=[0])
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


def phrase(list_of_words):
    """Creates a sentence linked with '_' from a list of words.
    Used for sentence recognition in word2vec models

    Parameters:
    -----------
    list : list of words forming the sentence
    """
    phr = ""
    for string in list_of_words:
        phr += string + "_"
    return phr


# def barycentre(list_of_words, model):
#     """Computes the embdeddings barycenter from a list of words
#     or sentences

#     Parameters:
#     -----------
#     list : list of words or sentences
#     model : the word2vec model containing the embeddings
#     """
#     barycenter = np.zeros(50, dtype=object)
#     for word in list_of_words:
#         barycenter = barycenter + model[word]
#     return barycenter / (len(list_of_words))


def compute_pc(array, npc=1):
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
    svd.fit(array)
    return svd.components_


def remove_pc(array, npc=1):
    """
    Remove the projection on the principal components

    Parameters:
    -----------
    X: X[i,:] is a data point
    npc: number of principal components to remove
    """
    print("Removing principal component...")
    pc = compute_pc(array, npc)
    if npc == 1:
        without_pc = array - array.dot(pc.transpose()) * pc
    else:
        without_pc = array - array.dot(pc.transpose()).dot(pc)
    return without_pc
