from __future__ import division
import numpy as np
from collections import defaultdict
import math
import json
from ..Polarization.polarization_functions import *
from ..GloVe.weights import *
from ..Axes.projection_functions import *
from ..Axes.axes_definition import *
from ..Axes.models import *


def get_counts(text, vocab):
    counts = {w: 0 for w in vocab}
    for split in text:
        count = 0
        prev = ""
        for w in split:
            if w == "":
                continue
            if w in vocab:
                counts[w] += 1
            if count > 0:
                bigram = prev + " " + w
                if bigram in vocab:
                    counts[bigram] += 1
            count += 1
            prev = w
    return counts


def log_odds(counts1, counts2, prior, zscore=True):
    # code from Dan Jurafsky
    # note: counts1 will be positive and counts2 will be negative

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())

    # since we use the sum of counts from the two groups as a prior, this is equivalent to a simple log odds ratio
    nprior = sum(prior.values())
    for word in prior.keys():
        if prior[word] == 0:
            delta[word] = 0
            continue
        l1 = float(counts1[word] + prior[word]) / (
            (n1 + nprior) - (counts1[word] + prior[word])
        )
        l2 = float(counts2[word] + prior[word]) / (
            (n2 + nprior) - (counts2[word] + prior[word])
        )
        sigmasquared[word] = 1 / (float(counts1[word]) + float(prior[word])) + 1 / (
            float(counts2[word]) + float(prior[word])
        )
        sigma[word] = math.sqrt(sigmasquared[word])
        delta[word] = math.log(l1) - math.log(l2)
        if zscore:
            delta[word] /= sigma[word]
    return delta


def get_log_odds_values(df_speeches, words2idx, party_1, party_2):
    dem_tweets, rep_tweets = (
        df_speeches[df_speeches["party"] == party_1],
        df_speeches[df_speeches["party"] == party_2],
    )

    # get counts
    counts1 = get_counts(rep_tweets["text"], words2idx)
    counts2 = get_counts(dem_tweets["text"], words2idx)
    prior = {}
    for k, v in counts1.items():
        prior[k] = v + counts2[k]

    # get log odds
    # note: we don't z-score because that makes the absolute values for large events significantly smaller than for smaller
    # events. however, z-scoring doesn't make a difference for our results, since we simply look at whether the log odds
    # are negative or positive (rather than their absolute value)
    delta = log_odds(counts1, counts2, prior, False)
    return prior, counts1, counts2, delta


def from_unigrams_to_bigrams(list_of_strings):
    return [
        list_of_strings[i] + " " + list_of_strings[i + 1]
        for i in range(len(list_of_strings) - 1)
    ]


def get_word_partisanship(
    df_speeches, year, party_1, party_2, bigram=False, with_parliament=True
):
    if with_parliament:
        with open("data/with parliament/vocabs/vocab_" + str(year) + ".json") as f:
            vocab = json.load(f)
    if not with_parliament:
        with open(
            "data/without parliament/vocabs/vocab_" + str(year) + "_WP.json"
        ) as f:
            vocab = json.load(f)

    if bigram:
        df_speeches["text"] = df_speeches["text"].apply(from_unigrams_to_bigrams)
        vocab = [vocab[i] + " " + vocab[i + 1] for i in range(len(vocab) - 1)]

    words2idx = {w: i for i, w in enumerate(vocab)}
    idx2words = {i: w for i, w in enumerate(vocab)}

    # get log odds
    prior, counts1, counts2, delta = get_log_odds_values(
        df_speeches, words2idx, party_1, party_2
    )

    # get counts for posterior, mutual information and chi square
    dem_tweets, rep_tweets = (
        df_speeches[df_speeches["party"] == "Lab"],
        df_speeches[df_speeches["party"] == "Con"],
    )
    dem_counts = get_user_token_counts(dem_tweets, words2idx)
    rep_counts = get_user_token_counts(rep_tweets, words2idx)
    dem_nonzero = set(dem_counts.nonzero()[0])
    rep_nonzero = set(rep_counts.nonzero()[0])
    dem_counts = dem_counts[
        np.array([(i in dem_nonzero) for i in range(dem_counts.shape[0])]), :
    ]  # filter users who did not use words from vocab
    rep_counts = rep_counts[
        np.array([(i in rep_nonzero) for i in range(rep_counts.shape[0])]), :
    ]

    # calculate posterior
    dem_q = get_party_q(dem_counts)
    rep_q = get_party_q(rep_counts)
    token_scores_rep = get_rho(dem_q, rep_q)

    # mutual information and chi square
    dem_no = dem_counts.shape[0]
    rep_no = rep_counts.shape[0]
    dem_t = get_token_user_counts(dem_counts)
    rep_t = get_token_user_counts(rep_counts)
    dem_not_t = dem_no - dem_t + 2  # because of add one smoothing
    rep_not_t = rep_no - rep_t + 2  # because of add one smoothing
    mutual_info = mutual_information(dem_t, rep_t, dem_not_t, rep_not_t, dem_no, rep_no)
    chi = chi_square(dem_t, rep_t, dem_not_t, rep_not_t, dem_no, rep_no)

    features = np.ndarray(
        (8, len(vocab))
    )  # prior, rep_count, dem_count, log odds, posterior, mutual information, chi square
    for w in vocab:
        features[0, words2idx[w]] = prior[w]
        features[1, words2idx[w]] = counts1[w]
        features[2, words2idx[w]] = counts2[w]
        features[3, words2idx[w]] = delta[w]
        features[4, words2idx[w]] = token_scores_rep[words2idx[w]]
        features[5, words2idx[w]] = mutual_info[words2idx[w]]
        features[6, words2idx[w]] = chi[words2idx[w]]

    return features, idx2words, words2idx


def cosine_with_axis_word(word: str, b, model_words):
    array_1 = model_words[word]
    array_2 = b

    cosine = np.dot(array_1, array_2.T) / (norm(array_1) * norm(array_2))

    return cosine


def cosine_with_axis_bigram(bigram: str, b, model_words):
    word_1, word_2 = bigram.split(" ")

    array_1 = (model_words[word_1] + model_words[word_2]) * 0.5
    array_2 = b

    cosine = np.dot(array_1, array_2.T) / (norm(array_1) * norm(array_2))

    return cosine


def get_quantiles(data, percentiles):
    """
    Get quantiles from a distribution.

    Parameters:
        data (array-like): The data.
        percentiles (array-like): The percentiles to compute (0-100).

    Returns:
        quantiles (array): The values at the specified percentiles.
    """
    return np.percentile(data, percentiles)


def filter_deltas(df, delta_low_percentile, delta_high_percentile):
    delta_percentiles = [delta_low_percentile, delta_high_percentile]

    quantiles_deltas = get_quantiles(df["deltas"], delta_percentiles)

    df = df.loc[
        (df["deltas"] < quantiles_deltas[0]) | (df["deltas"] > quantiles_deltas[1])
    ]

    return df


def get_quantiles(data, percentiles):
    """
    Compute quantiles for a given dataset and percentiles.

    :param data: Numerical data from which to calculate quantiles.
    :param percentiles: A list of percentiles to calculate for the data.
    :return: An array of quantiles corresponding to the specified percentiles.
    """
    return np.percentile(data, percentiles)


def partizan_words(
    left_side,
    right_side,
    year,
    gram="bigram",
    focus_on_companies=None,
    axis=None,
    percentiles_cos=[10, 90],
    percentiles_delta=[10, 90],
    force_i_lim=None,
    re_filter_cos=False,
    percentiles_refiltering_cos=[25, 75],
    with_parliament=True,
):
    sources = left_side + right_side

    s = str(year)[-2:]

    if s[0] == "1":
        i = eval(s[1])
    if s[0] == "2":
        i = eval("1" + s[1])

    if with_parliament:
        df_proj = pd.read_csv("data/with parliament/current_dataframes/df_BT.csv")
    if not with_parliament:
        df_proj = pd.read_csv("data/without parliament/current_dataframes/df_BT.csv")
        df_proj["party"], df_proj["Speaker"] = 0, 0

    df_par = df_proj.loc[
        df_proj["source"].isin(sources) | df_proj["party"].isin(sources)
    ]

    def change_year(old_year):
        if int(old_year) == 20110:
            return 2020
        if int(old_year) == 20111:
            return 2021
        if int(old_year) == 20112:
            return 2022
        if int(old_year) == 20113:
            return 2023
        else:
            return int(old_year)

    df_par["year"] = df_par["year"].apply(change_year)
    df_par = df_par.loc[df_par["year"] == year]

    df1 = df_par[df_par["source"] == "par"]
    df2 = df_par[df_par["source"] != "par"]

    # Define a function to translate newspaper source to party
    def translate_party(newspaper):
        """
        Translates newspaper sources to their corresponding political party.

        :param newspaper: The source to be translated.
        :return: The political party corresponding to the source.
        """
        if newspaper in left_side:
            return "Lab"
        if newspaper in right_side:
            return "Con"

    # Apply the translation function to assign parties based on sources
    df2["party"] = df2["source"].apply(translate_party)
    df2["Speaker"] = range(len(df2))

    # Combine the two DataFrames and reset index for continuity
    df_par = pd.concat([df1, df2]).reset_index(drop=True)

    if focus_on_companies:
        df_par = df_par.loc[df_par["class"].isin(focus_on_companies)]

    if axis is not None:
        quantiles = get_quantiles(df_par[f"cos axe {axis}"], percentiles_cos)
        df_par = df_par[
            (df_par[f"cos axe {axis}"] < quantiles[0])
            | (df_par[f"cos axe {axis}"] > quantiles[1])
        ]

    def phrase_to_tokens(phrase):
        word_list = phrase.strip("_").split("_")
        return word_list

    df_par["text"] = df_par["text"].apply(phrase_to_tokens)

    if gram == "bigram":
        partisanship_matrix, idx2words, words2idx = get_word_partisanship(
            df_par,
            year,
            "Lab",
            "Con",
            bigram=True,
            with_parliament=with_parliament,
        )
        deltas = partisanship_matrix[3, :]
        words = np.array(list(idx2words.values()), dtype=None)
        df_sorted_partisanships = pd.DataFrame(
            {"words": words, "deltas": deltas}
        ).sort_values(by="deltas", ascending=False)

        df = df_sorted_partisanships
        m = models_w[i]
        pos_a = filter_model(pos_1, m)
        neg_a = filter_model(neg_1, m)

        b = barycentre(pos_a, m) - barycentre(neg_a, m)
        df["cos"] = df["words"].apply(cosine_with_axis_bigram, b=b, model_words=m)
        df_sorted_partisanships = df
        df = filter_deltas(df, percentiles_delta[0], percentiles_delta[1])

    if gram == "unigram":
        partisanship_matrix, idx2words, words2idx = get_word_partisanship(
            df_par,
            year,
            "Lab",
            "Con",
            bigram=False,
            with_parliament=with_parliament,
        )
        deltas = partisanship_matrix[3, :]
        words = np.array(list(idx2words.values()), dtype=None)
        df_sorted_partisanships = pd.DataFrame(
            {"words": words, "deltas": deltas}
        ).sort_values(by="deltas", ascending=False)

        df = df_sorted_partisanships
        m = models_w[i]
        pos_a = filter_model(pos_1, m)
        neg_a = filter_model(neg_1, m)

        b = barycentre(pos_a, m) - barycentre(neg_a, m)
        df["cos"] = df["words"].apply(cosine_with_axis_word, b=b, model_words=m)
        df_sorted_partisanships = df
        df = filter_deltas(df, percentiles_delta[0], percentiles_delta[1])

    if re_filter_cos:
        quantiles = get_quantiles(df["cos"], percentiles_refiltering_cos)
        df = df[(df["cos"] < quantiles[0]) | (df["cos"] > quantiles[1])]

    return df.sort_values(by="deltas", ascending=False)
