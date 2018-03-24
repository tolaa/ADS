import numpy as np
import pandas as pd


def nan_processor(df, replacement_str=''):
    '''
    Take a dataframe and return one where all occurences of the replacement
    string have been replaced by `np.nan` and, consequently, all rows
    containing np.nan have been removed.

    Example with replacement_str='blah'
         A       B      C                   A     B    C
    --------------------------         ------------------
    0 |  0.5 |  0.3   | 'blah'         1 | 0.2 | 0.1 | 5
    1 |  0.2 |  0.1   |   5     -->    3 | 0.7 | 0.2 | 1
    2 |  0.1 | 'blah' |   3
    3 |  0.7 |  0.2   |   1

    :param df: Input data frame (pandas.DataFrame)
    :param replacement_str: string to find and replace by np.nan
    :returns: DataFrame where the occurences of replacement_str have been
        replaced by np.nan and subsequently all rows containing np.nan have
        been removed
    '''
    new_df = df.replace(to_replace=replacement_str, value=np.nan)
    new_df.dropna(inplace=True, axis=0)
    return new_df


def feature_cleaner(df, low=0.05, high=0.95):
    '''
    Take a dataframe where columns are all numerical and non-constant.
    For each feature, mark the values that are not between the given
    percentiles (low-high) as np.nan.
    Then, remove all rows containing np.nan.
    Finally, the columns must be scaled to have zero mean and unit variance
    (do this without sklearn).

    Example testdf:
            0     1     2
    ---------------------
    A |   0.1   0.2   0.1
    B |   5.0  10.0  20.0
    C |   0.2   0.3   0.5
    D |   0.3   0.2   0.7
    E |  -0.1  -0.2  -0.4
    F |   0.1   0.4   0.3
    G |  -0.5   0.3  -0.2
    H | -10.0   0.3   1.0

    Output of feature_cleaner(testdf, 0.01, 0.99):

                0         1         2
    ---------------------------------
    A |  0.191663 -0.956183 -0.515339
    C |  0.511101  0.239046  0.629858
    D |  0.830540 -0.956183  1.202457
    F |  0.191663  1.434274  0.057260
    G | -1.724967  0.239046 -1.374236

    :param df:      Input DataFrame (with numerical columns)
    :param low:     Lowest percentile  (0.0<low<1.0)
    :param high:    Highest percentile (low<high<1.0)
    :returns:       Scaled DataFrame where elements that are outside of the
                    desired percentiel range have been removed
    '''
    new_df = df.loc[:, df.apply(pd.Series.nunique) != 1]
    features = new_df.columns
    for feat in features:
        new_df[feat] = [i if np.nanpercentile(new_df[feat].values, int(low * 100)) < i < np.nanpercentile(
            new_df[feat].values, int(high * 100))
                        else np.nan for i in new_df[feat]]
    new_df.dropna(inplace=True, axis=0)
    new_df = new_df.apply(lambda x: (x - x.mean()) / x.std())
    return new_df


def get_feature(df):
    '''
    Take a dataframe where all columns are numerical and not constant.
    One of the column named "CLASS" is either 0 or 1.
    Within each class, for each feature compute the ratio (R) of the
    range over the variance (the range is the gap between the smallest
    and largest value).
    For each feature you now have two R (R_0 and R_1).
    For each feature, compute the ratio (say K) of the larger R to the smaller R.
    Return the name of the feature for which this last ratio K is largest.

    Test input
           A     B     C   CLASS
    ----------------------------
    0 |  0.1   0.2   0.1     0
    1 |  5.0  10.0  20.0     0
    2 |  0.2   0.3   0.5     1
    3 |  0.3   0.2   0.7     0
    4 |	-0.1  -0.2  -0.4     1
    5 |	 0.1   0.4   0.3     0
    6 |	-0.5   0.3  -0.2     0

    Output of get_feature(testdf) is 'C'

    :param df:  Input DataFrame (with numerical columns)
    :returns:   Name of the feature
    '''
    new_df = df.loc[:, df.apply(pd.Series.nunique) != 1]
    class_df = new_df.groupby('CLASS').max() - new_df.groupby('CLASS').min()
    col_dict = {}
    for col in class_df.columns:
        col_dict[col] = max(class_df[col]) / min(class_df[col])
    return max(col_dict, key=col_dict.get)


def one_hot_encode(label_to_encode, labels):
    '''
    Write a function that takes in a label to encode and a list of possible
    labels. It should return the label one-hot-encoded as a list of elements
    containing 0s and a unique 1 at the index corresponding to the matching
    label. Note that the input list of labels should contain unique elements.
    This function should raise a ValueError if the label_to_encode
    can not be found in list of labels.

    Examples:
    one_hot_encode("blue", ["blue", "red", "pink", "yellow"]) -> [1, 0, 0, 0]
    one_hot_encode("pink", ["blue", "red", "pink", "yellow"]) -> [0, 0, 1, 0]
    one_hot_encode("b", ["a", "b", "c", "d", "e"]) -> [0, 1, 0, 0, 0]

    :param label_to_encode: the label to encode
    :param labels: a list of all possible labels
    :return: a list of 0s and one 1
    :raise: ValueError
    '''
    if label_to_encode in labels and sorted(labels) == sorted(list(set(labels))):
        new_list = [1 if x == label_to_encode else 0 for x in labels]
        return new_list
    else:
        raise ValueError('Input valid labels')
