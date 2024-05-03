"""Functions to go from the texts embeddings to the 
cosine similarities with the defined axes."""

import ast
import warnings

from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
import numpy as np
from dask import dataframe as dd
from numpy.linalg import norm

warnings.filterwarnings("ignore")

from ..GloVe.weights import phrase


#################################################
# Transformation of the data for the projection #
#################################################


def tostring(l):
    """
    Converts a list to a string representation.

    Parameters:
    - list (List): The list to be converted.

    Returns:
    - str: A string representation of the input list.
    """
    return str(l)


def df_BT(df):
    """
    Process a dataframe to categorize and label rows based
    on thematic keywords related to major tech companies.

    The function creates a subset of the original dataframe
    based on the presence of specific company themes in the
    'keywords' column. It assigns a class label to each row
    indicating the company theme, then concatenates all subsets
    into a single dataframe.

    Parameters:
    - df (pd.DataFrame): A pandas DataFrame containing a
    'keywords' column where each entry is a list of keywords.

    Returns:
    - pd.DataFrame: A DataFrame containing only the entries
    that match one of the specified themes, with an added 'class'
      column indicating the theme.

    Notes:
    - The function modifies the input DataFrame in-place before
    making a copy to prevent changes to the original data.
    - It assumes that the 'theme' function is defined to extract
    the relevant themes based on keywords.
    """
    df_BT = df.copy()
    df_BT.reset_index(drop=True, inplace=True)
    df_BT["theme"] = df_BT["keywords"].apply(theme)
    df_BT["theme"] = df_BT["theme"].apply(tostring)

    df_BT_amazon = df_BT[df_BT["theme"].str.contains("amazon")]
    df_BT_facebook = df_BT[df_BT["theme"].str.contains("facebook")]
    df_BT_apple = df_BT[df_BT["theme"].str.contains("apple")]
    df_BT_google = df_BT[df_BT["theme"].str.contains("google")]
    df_BT_microsoft = df_BT[df_BT["theme"].str.contains("microsoft")]

    df_BT_amazon["class"] = "am"
    df_BT_facebook["class"] = "fb"
    df_BT_apple["class"] = "ap"
    df_BT_google["class"] = "go"
    df_BT_microsoft["class"] = "mi"

    df_BT = pd.concat(
        [
            df_BT_amazon,
            df_BT_facebook,
            df_BT_apple,
            df_BT_google,
            df_BT_microsoft,
        ]
    )
    return df_BT


def theme(keywords):
    """
    Identify the major tech company themes associated with a
    list of keywords.

    Parameters:
    - keywords (list of str): A list of keyword strings.

    Returns:
    - list: A list of unique company themes associated with the
      keywords provided. The themes include:
        'amazon', 'facebook', 'apple', 'google', and 'microsoft'.

    Each theme is determined by specific keywords that are
    associated with the respective company, including product
    names, services, and key personnel. The function returns a
    list of all themes that match the input keywords.
    """
    l = []
    for w in keywords:
        if w in [
            "amazon",
            "prime",
            "whole-foods",
            "zappos",
            "pillpack",
            "twitch",
            "audible",
            "goodreads",
            "imdb",
            "bezos",
            "jassy",
            "olsavsky",
            "alexander",
        ]:
            l.append("amazon")
        if w in [
            "meta",
            "facebook",
            "messenger",
            "instagram",
            "oculus",
            "whatsApp",
            "zuckerberg",
            "olivan",
            "clegg",
            "social-media",
        ]:
            l.append("facebook")
        if w in [
            "apple",
            "iphone",
            "iPad",
            "mac",
            "watch",
            "macbook",
            "ios",
            "airpods",
            "app-store",
            "itunes",
            "icloud",
            "beats",
            "siri",
            "shazam",
            "cook",
            "jobs",
        ]:
            l.append("apple")
        if w in [
            "google",
            "android",
            "chrome",
            "gmail",
            "maps",
            "playstore",
            "pixel",
            "waze",
            "youTube",
            "alphabet",
            "mandiant",
            "fitbit",
            "looker",
            "nest",
            "doubleclick",
            "page",
            "brin",
            "pichai",
            "kurian",
        ]:
            l.append("google")
        if w in [
            "microsoft",
            "windows",
            "cortana",
            "excel",
            "explorer",
            "office",
            "edge",
            "teams",
            "outlook",
            "powerpoint",
            "skype",
            "surface",
            "word",
            "xbox",
            "linkedIn",
            "github",
            "mojang",
            "gates",
            "nadella",
        ]:
            l.append("microsoft")
    return list(set(l))


def txt_to_model_sentences(data_loader, fichier: str):
    """Takes a txt file of sentence embeddings and translates it to a word2vec model format

    Parameters:
    -----------
    fichier : the txt file of sentence embeddings
    """
    # glove_file_sentences = datapath(fichier)
    word2vec_glove_file_sentences = get_tmpfile("format_word2vec.text")
    # glove2word2vec(fichier, word2vec_glove_file_sentences)
    data_loader.glove2word2vec(fichier, word2vec_glove_file_sentences)
    with open("data/" + fichier, "r") as file:
        data = file.readlines(fichier)
    data[0] = str(len(data) - 1) + " 50\n"
    with open("data/" + fichier, "w") as file:
        file.writelines(data)
    return KeyedVectors.load_word2vec_format(word2vec_glove_file_sentences)


def txt_to_model_words(data_loader, fichier: str):
    """Takes a txt file of word embeddings and translates it to a word2vec model format

    Parameters:
    -----------
    fichier : the txt file of word embeddings
    """
    # glove_file_sentences = datapath(fichier)
    word2vec_glove_file_sentences = get_tmpfile("format_word2vec.text")
    # glove2word2vec(fichier, word2vec_glove_file_sentences)
    data_loader.glove2word2vec(fichier, word2vec_glove_file_sentences)
    return KeyedVectors.load_word2vec_format(word2vec_glove_file_sentences)


#####################################
# Mathematical projection functions #
#####################################


def barycentre(list, model):
    """Computes the embdeddings barycenter from a list of words or sentences

    Parameters:
    -----------
    list : list of words or sentences
    model : the word2vec model containing the embeddings
    """
    if len(list) == 0:
        return None

    bar = np.zeros(50, dtype=object)
    for word in list:
        bar = bar + model[word]
    return bar / (len(list))


def filter_model(list, model_words):
    """Takes a list of words and a word2vec model, and removes
    the words from the list that do not appear in the model

    Parameters:
    -----------
    list : the list of words
    model_words : the word2vec word model
    """

    new_list = list.copy()

    for _ in range(10):
        for word in new_list:
            try:
                model_words[word]
            except:
                new_list.remove(word)
    return new_list


def proj_embedding_1(Wp):
    """
    Creates a copy of a nested list structure containing embeddings.

    This function iterates through a list of lists (typically embeddings or similar data structures)
    and duplicates its structure and contents. It is useful when a direct copy of the nested list's
    contents is needed for operations that should not affect the original data.

    Parameters:
    - Wp (list of list of float): A nested list where each sub-list contains numerical data
      (e.g., embeddings).

    Returns:
    - list of list of float: A new nested list with the same contents as the input.
    """
    l = [[] for _ in range(len(Wp))]  # Initialize a list of empty lists
    for i, sublist in enumerate(Wp):
        for item in sublist:
            l[i].append(item)
    return l


def projection_1D(pos_k: list, neg_k: list, model_words, model_sentences, df):
    """Returns the 2D projection matrix (Wp) of the sentences in
    df on the axes defined, and a dataframe containing the
    coordinates of these projections

    Parameters:
    -----------
    pos_k = the list of words forming the positive side of the first axis
    neg_k = the list of words forming the negative side of the first axis
    pos_l = the list of words forming the positive side of the second axis
    pos_l = the list of words forming the negative side of the second axis
    model_words : the word2vec word model used to get single words embeddings
    model_sentences : the word2vec sentence model used to get corresponding sentence embeddings
    df : the corpus dataframe
    """

    pos_a = filter_model(pos_k, model_words)
    neg_a = filter_model(neg_k, model_words)
    # pos_a = [model_words.most_similar(word)[i][0] for i in range(5) for word in pos_a]
    # neg_a = [model_words.most_similar(word)[i][0] for i in range(5) for word in neg_a]

    b1 = barycentre(pos_a, model_words) - barycentre(neg_a, model_words)
    b1 = b1 / np.linalg.norm(b1)

    proj_barycentres = [
        barycentre(pos_a, model_words),
        barycentre(neg_a, model_words),
    ]

    Wv = [i.tolist() for i in proj_barycentres]
    for i in df.index:
        sent = df["text"][i]
        Wv.append(model_sentences[sent].tolist())

    W = np.array(Wv)
    B = np.array([b1]).T
    B = B.astype("float64")
    BB = np.matmul(B.T, B)
    Bi = np.linalg.pinv(BB)
    Bii = np.matmul(Bi, B.T)
    Wp = np.matmul(Bii, W.T)

    return Wp


def projection_2D(
    pos_k: list,
    neg_k: list,
    pos_l: list,
    neg_l: list,
    model_words,
    model_sentences,
    df,
):
    """Returns the 2D projection matrix (Wp) of the sentences in
    df on the axes defined, and a dataframe containing the
    coordinates of these projections

    Parameters:
    -----------
    pos_k = the list of words forming the positive side of the first axis
    neg_k = the list of words forming the negative side of the first axis
    pos_l = the list of words forming the positive side of the second axis
    pos_l = the list of words forming the negative side of the second axis
    model_words : the word2vec word model used to get single words embeddings
    model_sentences : the word2vec sentence model used to get corresponding sentence embeddings
    df : the corpus dataframe
    """

    pos_a = filter_model(pos_k, model_words)
    neg_a = filter_model(neg_k, model_words)
    # pos_a = pos_a + [model_words.most_similar(word)[i][0] for i in range(5) for word in pos_a]
    # neg_a = neg_a + [model_words.most_similar(word)[i][0] for i in range(5) for word in neg_a]

    pos_b = filter_model(pos_l, model_words)
    neg_b = filter_model(neg_l, model_words)
    # pos_b = pos_b + [model_words.most_similar(word)[i][0] for i in range(5) for word in pos_b]
    # neg_b = neg_b + [model_words.most_similar(word)[i][0] for i in range(5) for word in neg_b]

    b1 = barycentre(pos_a, model_words) - barycentre(neg_a, model_words)
    b1 = b1 / np.linalg.norm(b1)
    b2 = barycentre(pos_b, model_words) - barycentre(neg_b, model_words)
    b2 = b2 / np.linalg.norm(b2)

    cosine = np.dot(b1, b2.T) / (norm(b1) * norm(b2))
    print("Cosine Similarity b1-b2:", cosine)

    proj_barycentres = [
        barycentre(pos_a, model_words),
        barycentre(neg_a, model_words),
        barycentre(pos_b, model_words),
        barycentre(neg_b, model_words),
    ]

    Wv = [i.tolist() for i in proj_barycentres]
    for i in df.index:
        sent = df["text"][i]
        Wv.append(model_sentences[sent].tolist())

    W = np.array(Wv)
    B = np.array([b1, b2]).T
    B = B.astype("float64")
    BB = np.matmul(B.T, B)
    Bi = np.linalg.pinv(BB)
    Bii = np.matmul(Bi, B.T)
    Wp = np.matmul(Bii, W.T)

    df["proj_embedding_x"] = proj_embedding_1(Wp)[0][4:]
    df["proj_embedding_y"] = proj_embedding_1(Wp)[1][4:]

    return (Wp, df)


def projection_3D(
    pos_k, neg_k, pos_l, neg_l, pos_m, neg_m, model_words, model_sentences, df
):
    """Returns the 3D projection matrix (Wp) of the sentences in
    df on the axes defined, and a dataframe containing the
      coordinates of these projections

    Parameters:
    -----------
    pos_k = the list of words forming the positive side of the first axis
    neg_k = the list of words forming the negative side of the first axis
    pos_l = the list of words forming the positive side of the second axis
    pos_l = the list of words forming the negative side of the second axis
    pos_m = the list of words forming the positive side of the third axis
    pos_m = the list of words forming the negative side of the third axis
    model_words : the word2vec word model used to get single words embeddings
    model_sentences : the word2vec sentence model used to get corresponding sentence embeddings
    df : the corpus dataframe
    """

    pos_a = filter_model(pos_k, model_words)
    neg_a = filter_model(neg_k, model_words)
    pos_a = [
        model_words.most_similar(word)[i][0]
        for i in range(5)
        for word in pos_a
    ]
    neg_a = [
        model_words.most_similar(word)[i][0]
        for i in range(5)
        for word in neg_a
    ]

    pos_b = filter_model(pos_l, model_words)
    neg_b = filter_model(neg_l, model_words)
    pos_b = [
        model_words.most_similar(word)[i][0]
        for i in range(5)
        for word in pos_b
    ]
    neg_b = [
        model_words.most_similar(word)[i][0]
        for i in range(5)
        for word in neg_b
    ]

    pos_c = filter_model(pos_m, model_words)
    neg_c = filter_model(neg_m, model_words)
    pos_c = [
        model_words.most_similar(word)[i][0]
        for i in range(5)
        for word in pos_c
    ]
    neg_c = [
        model_words.most_similar(word)[i][0]
        for i in range(5)
        for word in neg_c
    ]

    b1 = barycentre(pos_a, model_words) - barycentre(neg_a, model_words)
    b1 = b1 / np.linalg.norm(b1)
    b2 = barycentre(pos_b, model_words) - barycentre(neg_b, model_words)
    b2 = b2 / np.linalg.norm(b2)
    b3 = barycentre(pos_c, model_words) - barycentre(neg_c, model_words)
    b3 = b3 / np.linalg.norm(b3)

    cosine = np.dot(b1, b2.T) / (norm(b1) * norm(b2))
    print("Cosine Similarity b1-b2:", cosine)
    cosine = np.dot(b2, b3.T) / (norm(b2) * norm(b3))
    print("Cosine Similarity b2-b3:", cosine)
    cosine = np.dot(b1, b3.T) / (norm(b1) * norm(b3))
    print("Cosine Similarity b1-b3:", cosine)

    proj_barycentres = [
        barycentre(pos_a, model_words),
        barycentre(neg_a, model_words),
        barycentre(pos_b, model_words),
        barycentre(neg_b, model_words),
        barycentre(pos_c, model_words),
        barycentre(neg_c, model_words),
    ]

    Wv = [i.tolist() for i in proj_barycentres]
    for i in df.index:
        sent = df["text"][i]
        Wv.append(model_sentences[sent].tolist())

    W = np.array(Wv)
    B = np.array([b1, b2, b3]).T
    B = B.astype("float64")
    BB = np.matmul(B.T, B)
    Bi = np.linalg.pinv(BB)
    Bii = np.matmul(Bi, B.T)
    Wp = np.matmul(Bii, W.T)

    df["proj_embedding_x"] = proj_embedding_1(Wp)[0][6:]
    df["proj_embedding_y"] = proj_embedding_1(Wp)[1][6:]
    df["proj_embedding_z"] = proj_embedding_1(Wp)[2][6:]

    return (Wp, df)


def axis_vector(pos_1, neg_1, model_words):
    """
    Computes the directional vector for a semantic axis in a word embedding space based on
    positive and negative word sets.

    Parameters:
    - pos_1 (list of str): A list of words representing the positive pole of the semantic axis.
    - neg_1 (list of str): A list of words representing the negative pole of the semantic axis.
    - model_words (Word2Vec or similar): A word embedding model that contains embeddings for the
      words specified in pos_1 and neg_1.

    Returns:
    - numpy.ndarray: A vector representing the semantic axis defined by the positive and negative
      word sets.

    Notes:
    - The function relies on 'filter_model' to filter out words that do not exist in the model and
      'barycentre' to compute the centroid of the word embeddings.
    - The resulting vector can be used to project other embeddings onto this axis to measure
      their semantic similarity or difference relative to the defined axis.
    """
    pos_a = filter_model(pos_1, model_words)
    neg_a = filter_model(neg_1, model_words)

    b1 = barycentre(pos_a, model_words) - barycentre(neg_a, model_words)

    return b1


def cosine_with_axis(sentence: str, axis_v, model_sentences):
    """computes the cosine of a sentence with the given axis

    Parameters:
    -----------
    sentence : the sentence studied
    pos_1 : the list of words forming the positive side of the axis
    neg_1 : the list of words forming the negative side of the axis
    model_words : the word2vec word model used to get single words embeddings
    model_sentences : the word2vec sentence model used to get corresponding sentence embeddings
    """

    b1 = axis_v

    array_1 = model_sentences[sentence]
    array_2 = b1

    cosine_1 = np.dot(array_1, array_2.T) / (norm(array_1) * norm(array_2))

    return cosine_1


def bootstrap_mean(df, varname):
    """computes the bootstrap mean for the values of the column 'varnam' in df

    Parameters:
    -----------
    df : the dataframe used
    varnam : the column containing the values on which we compute the bootstrap mean
    """
    bootstrapped = df.sample(n=len(df), replace=True)
    return bootstrapped[varname].mean()


def intervals(t, digits=5):
    """computes the bootstrap 5% confidence intervals from the list of means t, up to 5 digits

    Parameters:
    -----------
    t : the list of bootstrap means computed previously
    """
    CI90 = np.percentile(t, [5, 95]).round(digits)
    return CI90


##################################
# Functions with graphic purpose #
##################################


def source_to_color(text):
    """
    Converts a newspaper source abbreviation into a corresponding color.

    Parameters:
    - text (str): The abbreviation of the newspaper source.

    Returns:
    - str: The color associated with the given newspaper source.

    Supported sources and their corresponding colors:
    - "DE" -> "yellow"
    - "DM" -> "green"
    - "GUA" -> "blue"
    - "MET" -> "pink"
    - "TE" -> "red"
    - "par" -> "grey"
    """
    if text == "DE":
        return "yellow"
    if text == "DM":
        return "green"
    if text == "GUA":
        return "blue"
    if text == "MET":
        return "pink"
    if text == "TE":
        return "red"
    if text == "par":
        return "grey"


def party_to_size(text, size_list):
    """
    Determines the size representation for a political party based on a predefined list.

    Parameters:
    - text (str): The abbreviation of the political party.
    - size_list (list of int): A list of sizes corresponding to each party.

    Returns:
    - int: The size corresponding to the given party from the size_list.

    The function assumes the order of parties in size_list is:
    Con, Lab, LibDem, SNP, DUP.
    """
    if text == "Con":
        return size_list[0]
    if text == "Lab":
        return size_list[1]
    if text == "LibDem":
        return size_list[2]
    if text == "SNP":
        return size_list[3]
    if text == "DUP":
        return size_list[4]


def party_to_color(text):
    """
    Converts a political party abbreviation into a corresponding color.

    Parameters:
    - text (str): The abbreviation of the political party.

    Returns:
    - str: The color associated with the given political party.

    Supported parties and their corresponding colors:
    - "Con" -> "red"
    - "Lab" -> "blue"
    - "LibDem" -> "turquoise"
    - "SNP" -> "pink"
    - "DUP" -> "violet"
    """
    if text == "Con":
        return "red"
    if text == "Lab":
        return "blue"
    if text == "LibDem":
        return "turquoise"
    if text == "SNP":
        return "pink"
    if text == "DUP":
        return "violet"


def change_format_year(year):
    """
    Corrects specific year representations that are incorrectly formatted.

    Parameters:
    - year (int): The year to be checked and potentially corrected.

    Returns:
    - int: The corrected year.

    This function specifically addresses year formats from a data processing error,
    where years were incorrectly recorded as 20110, 20111, 20112, 20113, which correspond
    to the years 2020, 2021, 2022, and 2023 respectively.
    """
    if year == 20110:
        year = 2020
    if year == 20111:
        year = 2021
    if year == 20112:
        year = 2022
    if year == 20113:
        year = 2023
    return year
