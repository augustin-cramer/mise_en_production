import pandas as pd
import numpy as np


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


def bootstrap(df_g_cos, df, source_column="source", axis=1):
    # Validation check for axis parameter
    if axis not in [1, 2]:
        raise ValueError("source parameter must be one of [1, 2]")

    df_g_cos["CI_" + str(axis) + "_sup"] = None
    df_g_cos["CI_" + str(axis) + "_inf"] = None

    for i in df_g_cos.index:
        year = df_g_cos["year"][i]
        source = df_g_cos[source_column][i]

        df_temp = df[df[source_column] == source]
        df_temp = df_temp[df_temp["year"] == year]

        t = [bootstrap_mean(df_temp, "cos axe " + str(axis)) for i in range(1001)]
        df_g_cos["CI_" + str(axis) + "_inf"][i] = intervals(t, digits=5)[0]
        df_g_cos["CI_" + str(axis) + "_sup"][i] = intervals(t, digits=5)[1]

    return df_g_cos
