from collections import Counter
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def zero_insert(x):
    '''
    Write a function that takes in a vector and returns a new vector where
    every element is separated by 4 consecutive zeros.

    Example:
    [4, 2, 1] --> [4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]

    :param x: input vector
    :type x:  numpy.array
    :returns: input vector with elements separated by 4 zeros
    :rtype:   numpy.array
    '''
    zero_values = np.zeros(4)
    if x != np.array([]) and x.ndim == 1:
        output_vector = np.array(x[0])
        for value in x[1:]:
            output_vector = np.hstack((output_vector,
                                       zero_values, np.array([value])))
        return output_vector.astype(x.dtype)
    else:
        return x


def vec_valid(vec):
    if vec.ndim > 1:
        raise ValueError('Please input a valid input vector')
    else:
        return None


def new_vec_valid(x, y):
    if x.size == 0 or y.size == 0:
        raise ValueError('Please input a valid input vector')
    else:
        return None


def return_closest(x, val):
    '''
    Write a function that takes in a vector and returns the value contained in
    the vector that is closest to a given value.

    Example:
    ([3, 4, 5], 2) --> 3

    :param x:   input vector
    :type x:    numpy.array of int/float
    :param val: input value
    :type val:  int/float
    :returns:   value from x closest to val
    :rtype:     int/float
    :raise:     ValueError
    '''
    if isinstance(x, np.ndarray) and x.ndim == 1:
        output_dict = {}
        for value in x:
            output_dict[value] = np.abs(value - val)
        closet_value = min(output_dict, key=output_dict.get)
        return closet_value
    else:
        raise ValueError('Please input a valid input vector')


def cauchy(x, y):
    '''
    Write a function that takes in two vectors
    and returns the associated Cauchy
    matrix with entries a_ij = 1/(x_i-y_j).

    Example:
    ([1, 2], [3, 4]) --> [[-1/2, -1/3], [-1, -1/2]]

    Note: the function should raise an error of type ValueError if there is a
    pair (i,j) such that x_i=y_j

    :param x: input vector
    :type x:  numpy.array of int/float
    :param y: input vector
    :type y:  numpy.array of int/float
    :returns: Cauchy matrix with entries 1/(x_i-y_j)
    :rtype:   numpy.array of float
    '''
    if isinstance(x, np.ndarray) and x.ndim == 1 \
            and isinstance(y, np.ndarray) and y.ndim == 1:
        x_shape = range(x.shape[0])
        y_shape = range(y.shape[0])
        cau = np.zeros((x.shape[0], y.shape[0]))
        for i in x_shape:
            for j in y_shape:
                if x[i] - y[j] != 0:
                    cau[i, j] = 1/(x[i] - y[j])
                else:
                    raise ValueError('Please input a valid vector')
        return cau
    else:
        raise ValueError('Please input a valid input vector')


def cos_func(x, y):
    num = np.dot(x, y)
    denom = np.linalg.norm(x, 2) * np.linalg.norm(y, 2)
    return num/denom


def most_similar(x, v_list):
    '''
    Write a function that takes in a vector x and a list of vectors and finds,
    in the list, the index of the vector that is most similar to x in the
    cosine-similarity sense.

    Example:
    ([1, 1], [[1, 0.9], [-1, 1]]) --> 0 (corresponding to [1,0.9])

    :param x:      input vector
    :type x:       numpy.array of int/float
    :param v_list: list of vectors
    :type v_list:  list of numpy.array
    :returns:      index of element in list that is closest to x in cosine-sim
    :rtype:        int
    '''
    vector_dict = {}
    if v_list:
        for vectors, index in zip(v_list, range(0, len(v_list))):
            vec_valid(vectors)
            vector_dict[index] = cos_func(x, vectors)
        similar_vector = max(vector_dict, key=vector_dict.get)
        return similar_vector
    else:
        return []


def gradient_descent(x_0, step, tol):
    '''
    Write a function that does a fixed-stepsize gradient descent on function f
    with gradient g and stops when the update has magnitude under a given
    tolerance level (i.e. when |xk-x(k-1)| < tol).
    Return a tuple with the position, the value of f at that position and the
    magnitude of the last update.
    h(x) = (x-1)^2 + exp(-x^2/2)
    f(x) = log(h(x))
    g(x) = (2(x-1) - x exp(-x^2/2)) / h(x)

    Example:
    (1.0, 0.1, 1e-3) --> approximately (1.2807, -0.6555, 0.0008)

    :param x_0:  initial point
    :type x_0:   float
    :param step: fixed step size
    :type step:  float
    :param tol:  tolerance for the magnitude of the update
    :type tol:   float
    :returns:    the position, the value at that position and the latest update
    :rtype:      tuple of three float
    '''
    h = lambda x: (x-1)**2 + np.exp(-(x**2)/2)
    f = lambda x: np.log(h(x))
    g = lambda x: ((2*(x-1))-(x*np.exp(-(x**2)/2)))/h(x)
    tol_1 = tol*1.5
    x = x_0
    while tol_1 > tol:
        x = x_0 - (step * g(x_0))
        tol_1 = abs(x - x_0)
        x_0 = x
    return tuple([x, f(x), tol_1])


def filter_rep(df):
    '''
    Write a function that takes a DataFrame with a colum `A` of integers and
    filters out the rows which contain the same integer as the row immediately
    above. Check that the index is right, use reset_index if necessary.

    Example:
        A   ...            A   ...
    ___________        ___________
    0 | 1 | ...        0 | 1 | ...
    1 | 1 | ...        1 | 0 | ...
    2 | 0 | ...  -->   2 | 5 | ...
    3 | 5 | ...        3 | 2 | ...
    4 | 5 | ...
    5 | 5 | ...
    6 | 2 | ...

    :param df: input data frame with a column `A`
    :type df:  pandas.DataFrame
    :returns:  a dataframe where rows have been filtered out
    :rtype:    pandas.DataFrame
    '''
    if 'A' in df.columns:
        df.drop_duplicates('A', inplace=True)
        df.reset_index(inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        return df
    else:
        return df


def subtract_row_mean(df):
    '''
    Given a DataFrame of numeric values, write a function to subtract the row
    mean from each element in the row.

    Example:
        A   B   C                A     B     C
    _____________         _____________________
    0 | 1 | 5 | 0    -->  0 | -1.0 | 3.0 | -2.0
    1 | 2 | 6 | 1         1 | -1.0 | 3.0 | -2.0

    :param df: input data frame
    :type df:  pandas.DataFrame
    :returns:  a dataframe where each row is centred
    :rtype:    pandas.DataFrame
    '''
    if df.empty:
        return df
    else:
        row_averages = df.mean(axis=1)
        row_averages = row_averages.to_frame()
        new_df = pd.merge(df, row_averages, left_index=True, right_index=True)
        for col in new_df.columns[:-1]:
            new_df[col] = new_df[col] - new_df[new_df.columns[-1]]
        new_df.drop(new_df.columns[-1], inplace=True, axis=1)
        return new_df


def all_unique_chars(string):
    '''
    Write a function to determine if a string is only made of unique
    characters and returns True if that's the case, False otherwise.
    Upper case and lower case should be considered as the same character.

    Example:
    "qwr#!" --> True, "q Qdf" --> False

    :param string: input string
    :type string:  string
    :returns:      true or false if string is made of unique characters
    :rtype:        bool
    '''
    string.replace(" ", "")
    if string:
        string_count = Counter(string.lower())
        frequency = string_count.most_common(1)[0][1]
        if frequency > 1:
            return False
        else:
            return True
    else:
        return True


def mat_update(sq_mat, val, output_list):
    cols_list = range(sq_mat.shape[1])
    for row in range(sq_mat.shape[0]):
        k = 0
        initial_value = sq_mat[row, k]
        while val <= initial_value and k < len(cols_list) - 1:
            if initial_value == val:
                # output_list.append(row)
                # output_list.append(k)
                output_list.append([row, k])
            k += 1
            if k < len(cols_list):
                initial_value = sq_mat[row, k]
            else:
                initial_value = val + 1


def find_element(sq_mat, val):
    '''
    Write a function that takes a square matrix of integers and returns the
    position (i,j) of a value. The position should be returned as a list of two
    integers. If the value is present multiple times, a single valid list
    should be returned.
    The matrix is structured in the following way:
    - each row has strictly decreasing values with the column index increasing
    - each column has strictly decreasing values with the row index increasing
    The following matrix is an example:

    Example:
    mat = [ [10, 7, 5],
            [ 9, 4, 2],
            [ 5, 2, 1] ]
    find_element(mat, 4) --> [1, 1]

    The function should raise an exception ValueError if the value isn't found.
    The time complexity of the function should be linear in the number of rows.

    :param sq_mat: the square input matrix with decreasing rows and columns
    :type sq_mat:  numpy.array of int
    :param val:    the value to be found in the matrix
    :type val:     int
    :returns:      the position of the value in the matrix
    :rtype:        list of int
    '''
    if sq_mat.shape[0] > 0 and val in sq_mat:
        output_list = []
        mat_update(sq_mat, val, output_list)
        # return output_list[0] if len(output_list) == 1 else set(output_list)
        for pos in output_list:
            return pos
    else:
        raise ValueError('Please input a valid vector')


def filter_matrix(mat):
    '''
    Write a function that takes an n x p matrix of integers and sets the rows
    and columns of every zero-entry to zero.

    Example:
    [ [1, 2, 3, 1],        [ [0, 2, 0, 1],
      [5, 2, 0, 2],   -->    [0, 0, 0, 0],
      [0, 1, 3, 3] ]         [0, 0, 0, 0] ]

    The complexity of the function should be linear in n and p.

    :param mat: input matrix
    :type mat:  numpy.array of int
    :returns:   a matrix where rows and columns of zero entries in mat are zero
    :retype:    numpy.array
    '''
    new_mat = mat.copy()
    output_list = []
    for index, rows in enumerate(new_mat):
        for ind, j in enumerate(rows):
            if j == 0:
                output_list.append([index, ind])
    for pos in output_list:
        new_mat[pos[0], :] = 0
        new_mat[:, pos[1]] = 0
    return new_mat


def largest_sum(intlist):
    '''
    Write a function that takes in a list of integers, finds the continguous
    sublist with at least one element with the largest sum and returns the sum.

    Example:
    [-1, 2, 2] --> 4 (corresponding to [2, 2])

    Time complexity target: linear in the number of integers in the list.

    :param intlist: input list of integers
    :type intlist:  list of int
    :returns:       the largest sum
    :rtype:         int
    '''
    if intlist:
        sum_list = []
        listlength = len(intlist)
        indices = list(range(listlength+1))
        for i, j in itertools.combinations(indices, 2):
            sum_list.append(sum(intlist[i:j]))
        return max(sum_list)
    else:
        return 0


def pairprod(intlist, val):
    '''
    Write a function that takes in a list of integers and returns all unique
    pairs of elements whose product is equal to a given value. The pairs should
    all be of the form (i, j) with i<=j. The ordering of the pairs does not
    matter.

    Example:
    ([3, 5, 1, 2, 3, 6], 6) --> [(2, 3), (1, 6)]

    Complexity target: subquadratic

    :param intlist: input list of integers
    :type intlist:  list of int
    :param val:     given value products will be compared to
    :type val:      int
    :returns:       pairs of elements such that the product of corresponding
                    entries matches the value val
    :rtype:         list of tuple
    '''
    output_list = []
    for i, j in enumerate(intlist):
        for val_1, val_2 in enumerate(intlist):
            if i != val_1:
                if j * val_2 == val:
                    output_list.append(tuple(sorted((j, val_2))))
    return list(set(output_list))


def draw_co2_plot():
    '''
    Here is some chemistry data

      Time (decade): 0, 1, 2, 3, 4, 5, 6
      CO2 concentration (ppm): 250, 265, 272, 260, 300, 320, 389

    Create a line graph of CO2 versus time, the line should be a blue dashed
    line. Add a title and axis titles to the plot.
    '''
    time = range(0, 7)
    co2 = [250, 265, 272, 260, 300, 320, 389]
    plt.plot(time, co2, '--', color='blue')
    plt.title('CO2 Concentration', fontsize=12)
    plt.xlabel('Time')
    plt.ylabel('CO2')
    plt.show()


def draw_equations_plot():
    '''
    Plot the following lines on the same plot

      y=cos(x) coloured in red with dashed lines
      y=x^2 coloured in blue with linewidth 3
      y=exp(-x^2) coloured in black

    Add a legend, title for the x-axis and a title to the curve, the x-axis
    should range from -4 to 4 and the y axis should range from 0 to 2. The
    figure should have a size of 8x6 inches. The y-axis should be centered to
    the middle of your plot.
    '''
    x = np.linspace(-4, 4, 100)
    y_vec = np.cos(x)
    z_vec = x**2
    w_vec = np.exp(-x**2)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_vec, '--', color='red', label='Cos')
    plt.plot(x, z_vec, color='blue', linewidth=3, label='Squared')
    plt.plot(x, w_vec, color='black', label='Exponential')
    plt.xlim(-4, 4)
    plt.ylim(0, 2)
    plt.legend(loc=4)
    plt.show()
