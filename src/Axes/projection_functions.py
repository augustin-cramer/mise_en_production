from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from dask import dataframe as dd
import ast
from numpy.linalg import norm
from ..GloVe.weights import *
import warnings
import os

warnings.filterwarnings("ignore")

#################################################
# Transformation of the data for the projection #
#################################################


def tostring(list):
    return str(list)


def df_BT(df):
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


def txt_to_model_sentences(fichier: str):
    """Takes a txt file of sentence embeddings and translates it to a word2vec model format

    Parameters:
    -----------
    fichier : the txt file of sentence embeddings
    """
    # glove_file_sentences = datapath(fichier)
    word2vec_glove_file_sentences = get_tmpfile("format_word2vec.text")
    glove2word2vec(fichier, word2vec_glove_file_sentences)
    with open(word2vec_glove_file_sentences, "r") as file:
        data = file.readlines()
        data[0] = str(len(data) - 1) + " 50\n"
    with open(word2vec_glove_file_sentences, "w") as file:
        file.writelines(data)
    return KeyedVectors.load_word2vec_format(word2vec_glove_file_sentences)


def txt_to_model_words(fichier: str):
    """Takes a txt file of word embeddings and translates it to a word2vec model format

    Parameters:
    -----------
    fichier : the txt file of word embeddings
    """
    # glove_file_sentences = datapath(fichier)
    word2vec_glove_file_sentences = get_tmpfile("format_word2vec.text")
    glove2word2vec(fichier, word2vec_glove_file_sentences)
    return KeyedVectors.load_word2vec_format(word2vec_glove_file_sentences)


def open_to_project(path: str, year):
    """Takes a csv dataframe and formats it in order to facilitate the following manipulations
    Parameters:
    -----------
    path : the path to the csv dataframe
    year : the year corresponding to the dataframe
    """
    df = dd.read_csv(path, assume_missing=True)
    df = df.compute()
    df["text"] = df["text"].map(ast.literal_eval)
    df["year"] = year
    df["text"] = df["text"].apply(phrase)
    return df


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
    """Takes a list of words and a word2vec model, and removes the words from the list that do not appear in the model

    Parameters:
    -----------
    list : the list of words
    model_words : the word2vec word model
    """

    new_list = list.copy()

    for i in range(10):
        for word in new_list:
            try:
                model_words[word]
            except:
                new_list.remove(word)
    return new_list


def proj_embedding_1(Wp):
    l = [[] for i in range(len(Wp))]
    for i in range(len(Wp)):
        for j in range(len(Wp[i])):
            l[i].append(Wp[i][j])
    return l


def projection_1D(pos_k: list, neg_k: list, model_words, model_sentences, df):
    """Returns the 2D projection matrix (Wp) of the sentences in df on the axes defined, and a dataframe containing the coordinates of these projections

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

    # df['proj_embedding_x'] = proj_embedding_1(Wp)[0][4:]

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
    """Returns the 2D projection matrix (Wp) of the sentences in df on the axes defined, and a dataframe containing the coordinates of these projections

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
    """Returns the 3D projection matrix (Wp) of the sentences in df on the axes defined, and a dataframe containing the coordinates of these projections

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
    pos_a = [model_words.most_similar(word)[i][0] for i in range(5) for word in pos_a]
    neg_a = [model_words.most_similar(word)[i][0] for i in range(5) for word in neg_a]

    pos_b = filter_model(pos_l, model_words)
    neg_b = filter_model(neg_l, model_words)
    pos_b = [model_words.most_similar(word)[i][0] for i in range(5) for word in pos_b]
    neg_b = [model_words.most_similar(word)[i][0] for i in range(5) for word in neg_b]

    pos_c = filter_model(pos_m, model_words)
    neg_c = filter_model(neg_m, model_words)
    pos_c = [model_words.most_similar(word)[i][0] for i in range(5) for word in pos_c]
    neg_c = [model_words.most_similar(word)[i][0] for i in range(5) for word in neg_c]

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
    if year == 20110:
        year = 2020
    if year == 20111:
        year = 2021
    if year == 20112:
        year = 2022
    if year == 20113:
        year = 2023
    return year
