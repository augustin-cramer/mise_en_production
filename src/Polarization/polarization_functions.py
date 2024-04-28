""" This script contains the functions to compute the 
polarization of a corpus given two parties, following the 
methods of Gentzkow and al."""

from __future__ import division
from collections import Counter
import sys
import gc
import json
import random
import math
import scipy.sparse as sp
import numpy as np

sys.path.append("..")

RNG = random.Random() 

def get_user_token_counts(df_speeches, vocab):
    """
    Generates a sparse matrix of token counts per user from speeches.

    Parameters:
    - df_speeches (DataFrame): DataFrame containing speeches with a 'Speaker' and 'text' columns.
    - vocab (dict): Dictionary mapping words to indices.

    Returns:
    - csr_matrix: Sparse matrix of word counts per speaker.
    """
    # user-based
    speakers = df_speeches.groupby("Speaker")
    row_idx = []
    col_idx = []
    data = []
    for (
        group_idx,
        (u, group),
    ) in enumerate(speakers):
        word_indices = []
        for split in group["text"]:
            count = 0
            prev = ""
            for w in split:
                if w == "":
                    continue
                if w in vocab:
                    word_indices.append(vocab[w])
                if count > 0:
                    bigram = prev + " " + w
                    if bigram in vocab:
                        word_indices.append(vocab[bigram])
                count += 1
                prev = w
        for k, v in Counter(word_indices).items():
            col_idx.append(group_idx)
            row_idx.append(k)
            data.append(v)
    return sp.csr_matrix(
        (data, (col_idx, row_idx)), shape=(len(speakers), len(vocab))
    )


def get_party_q(party_counts, exclude_user_id=None):
    """
    Calculates the normalized token frequency for a party.

    Parameters:
    - party_counts (csr_matrix): Sparse matrix of token counts for a party.
    - exclude_user_id (int, optional): User ID to exclude from calculations.

    Returns:
    - np.ndarray: Normalized token frequencies.
    """
    user_sum = party_counts.sum(axis=0)
    if exclude_user_id:
        user_sum -= party_counts[exclude_user_id, :]
    total_sum = user_sum.sum()
    return user_sum / total_sum


def get_rho(dem_q, rep_q):
    """
    Calculates the proportion of Republican tokens to the total tokens.

    Parameters:
    - dem_q (np.ndarray): Democratic token frequencies.
    - rep_q (np.ndarray): Republican token frequencies.

    Returns:
    - np.ndarray: Proportion of Republican tokens.
    """
    return (rep_q / (dem_q + rep_q)).transpose()


def get_token_user_counts(party_counts):
    """
    Counts the number of users that use each token.

    Parameters:
    - party_counts (csr_matrix): Sparse matrix of token counts for a party.

    Returns:
    - np.ndarray: Count of users using each term, with smoothing.
    """
    no_tokens = party_counts.shape[1]
    nonzero = sp.find(party_counts)[:2]
    user_t_counts = Counter(nonzero[1])  # number of users using each term
    party_t = np.ones(no_tokens)  # add one smoothing
    for k, v in user_t_counts.items():
        party_t[k] += v
    return party_t


def mutual_information(dem_t, rep_t, dem_not_t, rep_not_t, dem_no, rep_no):
    """
    Computes the mutual information between token usage and party affiliation.

    Parameters:
    - dem_t, rep_t, dem_not_t, rep_not_t (np.ndarray): Token counts for and against each party.
    - dem_no, rep_no (int): Number of users in Democratic and Republican parties.

    Returns:
    - np.ndarray: Mutual information values for tokens.
    """
    no_users = dem_no + rep_no
    print(no_users)
    all_t = dem_t + rep_t
    all_not_t = no_users - all_t + 4
    mi_dem_t = dem_t * np.log2(no_users * (dem_t / (all_t * dem_no)))
    mi_dem_not_t = dem_not_t * np.log2(
        no_users * (dem_not_t / (all_not_t * dem_no))
    )
    mi_rep_t = rep_t * np.log2(no_users * (rep_t / (all_t * rep_no)))
    mi_rep_not_t = rep_not_t * np.log2(
        no_users * (rep_not_t / (all_not_t * rep_no))
    )
    return (
        1 / no_users * (mi_dem_t + mi_dem_not_t + mi_rep_t + mi_rep_not_t)
    ).transpose()[:, np.newaxis]


def chi_square(dem_t, rep_t, dem_not_t, rep_not_t, dem_no, rep_no):
    """
    Calculates the chi-square statistic for the association between tokens and party.

    Parameters:
    - dem_t, rep_t, dem_not_t, rep_not_t (np.ndarray): Token counts for and against each party.
    - dem_no, rep_no (int): Number of users in Democratic and Republican parties.

    Returns:
    - np.ndarray: Chi-square values for tokens.
    """
    no_users = dem_no + rep_no
    all_t = dem_t + rep_t
    all_not_t = no_users - all_t + 4
    chi_enum = no_users * (dem_t * rep_not_t - dem_not_t * rep_t) ** 2
    chi_denom = all_t * all_not_t * (dem_t + dem_not_t) * (rep_t + rep_not_t)
    return (chi_enum / chi_denom).transpose()[:, np.newaxis]


def calculate_polarization(
    dem_counts, rep_counts, measure="posterior", leaveout=True
):
    """
    Calculates polarization between two parties using specified statistical measures.

    Parameters:
    - dem_counts, rep_counts (csr_matrix): Token counts for Democratic and Republican parties.
    - measure (str): Statistical measure to use ('posterior', 'mutual_information', or 'chi_square').
    - leaveout (bool): Whether to use leave-one-out strategy for calculation.

    Returns:
    - float: Polarization score based on the specified measure.
    """
    dem_user_total = dem_counts.sum(axis=1)
    rep_user_total = rep_counts.sum(axis=1)

    dem_user_distr = (sp.diags(1 / dem_user_total.A.ravel())).dot(
        dem_counts
    )  # get row-wise distributions
    rep_user_distr = (sp.diags(1 / rep_user_total.A.ravel())).dot(rep_counts)
    dem_no = dem_counts.shape[0]
    rep_no = rep_counts.shape[0]
    assert set(dem_user_total.nonzero()[0]) == set(
        range(dem_no)
    )  # make sure there are no zero rows
    assert set(rep_user_total.nonzero()[0]) == set(
        range(rep_no)
    )  # make sure there are no zero rows
    if measure not in ("posterior", "mutual_information", "chi_square"):
        print("invalid method")
        return
    dem_q = get_party_q(dem_counts)
    rep_q = get_party_q(rep_counts)
    dem_t = get_token_user_counts(dem_counts)
    rep_t = get_token_user_counts(rep_counts)
    dem_not_t = dem_no - dem_t + 2  # because of add one smoothing
    rep_not_t = rep_no - rep_t + 2  # because of add one smoothing
    func = (
        mutual_information if measure == "mutual_information" else chi_square
    )

    # apply measures via leave-out
    dem_addup = 0
    rep_addup = 0
    dem_leaveout_no = dem_no - 1
    rep_leaveout_no = rep_no - 1

    for i in range(dem_no):
        if measure == "posterior":
            dem_leaveout_q = get_party_q(dem_counts, i)
            token_scores_dem = 1.0 - get_rho(dem_leaveout_q, rep_q)
        else:
            dem_leaveout_t = dem_t.copy()
            excl_user_terms = sp.find(dem_counts[i, :])[1]
            for term_idx in excl_user_terms:
                dem_leaveout_t[term_idx] -= 1
            dem_leaveout_not_t = dem_leaveout_no - dem_leaveout_t + 2
            token_scores_dem = func(
                dem_leaveout_t,
                rep_t,
                dem_leaveout_not_t,
                rep_not_t,
                dem_leaveout_no,
                rep_no,
            )
        dem_addup += dem_user_distr[i, :].dot(token_scores_dem)[0, 0]

    for i in range(rep_no):
        if measure == "posterior":
            rep_leaveout_q = get_party_q(rep_counts, i)
            token_scores_rep = get_rho(dem_q, rep_leaveout_q)
        else:
            rep_leaveout_t = rep_t.copy()
            excl_user_terms = sp.find(rep_counts[i, :])[1]
            for term_idx in excl_user_terms:
                rep_leaveout_t[term_idx] -= 1
            rep_leaveout_not_t = rep_leaveout_no - rep_leaveout_t + 2
            token_scores_rep = func(
                dem_t,
                rep_leaveout_t,
                dem_not_t,
                rep_leaveout_not_t,
                dem_no,
                rep_leaveout_no,
            )
        rep_addup += rep_user_distr[i, :].dot(token_scores_rep)[0, 0]

    rep_val = 1 / rep_no * rep_addup
    dem_val = 1 / dem_no * dem_addup
    return 1 / 2 * (dem_val + rep_val)


def get_values(
    df,
    year,
    party_1,
    party_2,
    token_partisanship_measure="posterior",
    leaveout=True,
    default_score=0.5,
    with_parliament=True,
):
    """
    Computes polarization values for given parties within a specific dataset.

    Parameters:
    - df (DataFrame): Data containing parties' data.
    - year (int): Year of data.
    - party_1, party_2 (str): The parties to analyze.
    - token_partisanship_measure (str): The method to measure token partisanship.
    - leaveout (bool): Whether to use leave-out strategy in calculation.
    - default_score (float): Default partisanship score if not enough data.
    - with_parliament (bool): If the analysis includes parliament data.

    Returns:
    - tuple: Actual polarization value, random polarization value, and total user count.
    """

    dem_tweets, rep_tweets = (
        df[df["party"] == party_1],
        df[df["party"] == party_2],
    )  # get partisan tweets

    if with_parliament:
        with open(
            "data/with parliament/vocabs/vocab_" + str(year) + ".json"
        ) as f:
            vocab = json.load(f)
    if not with_parliament:
        with open(
            "data/without parliament/vocabs/vocab_" + str(year) + "_WP.json"
        ) as f:
            vocab = json.load(f)

    # get vocab
    vocab = {w: i for i, w in enumerate(vocab)}

    dem_counts = get_user_token_counts(dem_tweets, vocab)
    rep_counts = get_user_token_counts(rep_tweets, vocab)

    dem_user_len = dem_counts.shape[0]
    rep_user_len = rep_counts.shape[0]

    if dem_user_len < 10 or rep_user_len < 10:
        return (
            default_score,
            default_score,
            dem_user_len + rep_user_len,
        )  # return these values when there is not enough data to make predictions on
    del dem_tweets
    del rep_tweets
    gc.collect()

    # make the prior neutral (i.e. make sure there are the same number of Rep and Dem users)
    dem_user_len = dem_counts.shape[0]
    rep_user_len = rep_counts.shape[0]
    if dem_user_len > rep_user_len:
        dem_subset = np.array(RNG.sample(range(dem_user_len), rep_user_len))
        dem_counts = dem_counts[dem_subset, :]
        dem_user_len = dem_counts.shape[0]
    elif rep_user_len > dem_user_len:
        rep_subset = np.array(RNG.sample(range(rep_user_len), dem_user_len))
        rep_counts = rep_counts[rep_subset, :]
        rep_user_len = rep_counts.shape[0]
    assert dem_user_len == rep_user_len

    all_counts = sp.vstack([dem_counts, rep_counts])

    wordcounts = all_counts.nonzero()[1]

    # filter words used by fewer than 2 people
    all_counts = all_counts[
        :,
        np.array(
            [
                (np.count_nonzero(wordcounts == i) > 1)
                for i in range(all_counts.shape[1])
            ]
        ),
    ]

    dem_counts = all_counts[:dem_user_len, :]
    rep_counts = all_counts[dem_user_len:, :]
    del wordcounts
    del all_counts
    gc.collect()

    dem_nonzero = set(dem_counts.nonzero()[0])
    rep_nonzero = set(rep_counts.nonzero()[0])
    dem_counts = dem_counts[
        np.array([(i in dem_nonzero) for i in range(dem_counts.shape[0])]), :
    ]  # filter users who did not use words from vocab
    rep_counts = rep_counts[
        np.array([(i in rep_nonzero) for i in range(rep_counts.shape[0])]), :
    ]
    del dem_nonzero
    del rep_nonzero
    gc.collect()

    actual_val = calculate_polarization(
        dem_counts, rep_counts, token_partisanship_measure, leaveout
    )

    all_counts = sp.vstack([dem_counts, rep_counts])
    del dem_counts
    del rep_counts
    gc.collect()

    index = np.arange(all_counts.shape[0])
    RNG.shuffle(index)
    all_counts = all_counts[index, :]

    random_val = calculate_polarization(
        all_counts[:dem_user_len, :],
        all_counts[dem_user_len:, :],
        token_partisanship_measure,
        leaveout,
    )

    sys.stdout.flush()
    del all_counts
    gc.collect()

    return actual_val, random_val, dem_user_len + rep_user_len


def compute_polarization_and_CI(df, year, party_1, party_2):
    """
    Computes polarization and confidence intervals for the given dataset and parties.

    Parameters:
    - df (DataFrame): Dataset to analyze.
    - year (int): Year of the dataset.
    - party_1, party_2 (str): Parties between which polarization is measured.

    Returns:
    - tuple: Real polarization value, random polarization, and confidence intervals.
    """
    tau = len(df)
    pi_s = []
    sqrt_tau_s = []
    random_pi_s = []

    for _ in range(100):
        sub_sample_k = df.sample(frac=0.1, replace=False, random_state=1)
        values = get_values(
            sub_sample_k,
            year,
            party_1,
            party_2,
            token_partisanship_measure="posterior",
            leaveout=True,
            default_score=0.5,
        )
        pol_k = values[0]
        random_pol = values[1]
        tau_k = len(sub_sample_k)

        pi_s.append(pol_k)
        random_pi_s.append(random_pol)
        sqrt_tau_s.append(math.sqrt(tau_k))

    # Real part

    pi_s = np.array(pi_s)
    sqrt_tau_s = np.array(sqrt_tau_s)
    means = np.array([np.mean(pi_s)] * len(pi_s))
    Q_s = np.multiply(sqrt_tau_s, pi_s - means)

    percentiles_real = np.percentile(Q_s, [10, 90]).round(5)

    # Random part

    random_pi_s = np.array(random_pi_s)
    random_means = np.array([np.mean(random_pi_s)] * len(random_pi_s))
    random_Q_s = np.multiply(sqrt_tau_s, random_pi_s - random_means)

    percentiles_random = np.percentile(random_Q_s, [10, 90]).round(5)

    values = get_values(
        df,
        year,
        party_1,
        party_2,
        token_partisanship_measure="posterior",
        leaveout=True,
        default_score=0.5,
    )

    real_pi = values[0]
    random_pol = values[1]

    CI_low_real = real_pi - (percentiles_real[1] / (math.sqrt(tau)))
    CI_high_real = real_pi - (percentiles_real[0] / (math.sqrt(tau)))

    CI_low_random = random_pol - (percentiles_random[1] / (math.sqrt(tau)))
    CI_high_random = random_pol - (percentiles_random[0] / (math.sqrt(tau)))

    return (
        real_pi,
        random_pol,
        CI_low_real,
        CI_high_real,
        CI_low_random,
        CI_high_random,
    )
