#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 01:00:45 2018

@author: alexskrn
"""

# 1.  Import statements for necessary package(s)
import numpy as np

# Suppress scientific notation in output
np.set_printoptions(suppress=True,
                    formatter={"float_kind": "{:0.2f}".format})


# 2.  Create 3 different numpy arrays with at least 30 items each
def get_ages():
    """Return a numpy array of ages of some group of people."""
    ages = np.array([30, 23, 27, 20, 24, 35, 32, 27, 25, 28, 26,
                     31, 32, 27, 35, 20, 22, 29, 33, 21, 20, 22,
                     29, 27, 24, 25, 25, 27, 27, 21, 27, 29, 24,
                     31, 28, 22, 29, 33, 29, 22, 25, 27, 29, 22,
                     30, 35, 25, 34, 28, 25, 23, 23, 27, 27, 23,
                     29, 29, 27]
                    )
    return ages


def get_movie_data():
    """Return a 10 x 4 np array containing some movie data.

    Columns: "Gross Revenue", "Rotten Rating", "IMBD Rating", "Budget"
    """
    movie_data = np.array([[2782300000, 83, 8.0, 237.0],
                           [2185400000, 88, 7.6, 200.0],
                           [1341500000, 96, 8.1, 125.0],
                           [1108600000, 92, 7.8, 200.0],
                           [1084400000, 87, 8.6, 250.0],
                           [1027000000, 57, 6.5, 115.0],
                           [1017000000, 65, 8.1, 180.0],
                           [1004600000, 94, 9.0, 185.0],
                           [974800000, 80, 7.3, 125.0],
                           [969900000, 92, 8.0, 63.0]
                           ]
                          )
    return movie_data


def get_usa_data():
    """Return a 13 x 4 numpy array with some USA data with missing values.

    Columns: "population", "# motor vehicles", "gas price", "co2 emission"
    """
    usa_data = np.array([[282162411, 221475173, 0.47, 5713560],
                         [284968955, 230428326, None, 5601405],
                         [287625193, 229619979, 0.4, 5650950],
                         [290107933, 231389998, None, 5681664],
                         [292805298, 237242615, 0.54, 5790765],
                         [295516599, 241193974, None, 5826394],
                         [298379912, 244165686, 0.63, 5737616],
                         [301231207, 247264605, None, 5828696],
                         [304093966, 248164738, 0.56, 5656839],
                         [306771529, 246282887, None, 5311840],
                         [309326225, 242042185, 0.76, 5433057],
                         [311587816, 244774957, None, None],
                         [313914040, None, 0.97, None]
                         ]
                        )

    return usa_data


# 3.  Write function(s) that remove outliers in the first array
def remove_outliers(array_like):
    """Return a new numpy array with removed outliers.

    array_like: a one-dimensional array-like object (numpy or list)
    Outliers are values beyond 2+ standard deviations from the mean.
    """
    # Make a copy of the input array and if the input is a usual Python
    # list convert it into a numpy array
    new_array = np.copy(array_like)

    LimitHi = np.mean(new_array) + 2*np.std(new_array)
    LimitLo = np.mean(new_array) - 2*np.std(new_array)
    FlagGood = (new_array >= LimitLo) & (new_array <= LimitHi)
    new_array = new_array[FlagGood]

    return new_array


# 4.  Write function(s) that replace outliers in the second array
def replace_outliers_median(array_like):
    """Return a new numpy array with outliers replaced with median values.

    array_like: a multi-dimensional array-like object (numpy or list).
    Outliers are 2+ standard deviations from the mean of column values.
    """
    # Make a copy of the input array and if the input is a usual Python
    # list convert it into a numpy array
    new_array = np.copy(array_like)

    # Transpose the array to operate on columns like on rows
    for col in new_array.T:
        LimitHi = np.mean(col) + 2*np.std(col)
        LimitLo = np.mean(col) - 2*np.std(col)
        FlagBad = (col < LimitLo) | (col > LimitHi)
        if col[FlagBad].size > 0:  # If there are outliers
            col[FlagBad] = np.median(col)

    return new_array


# 5.  Write function(s) that fill in missing values in the third array
def is_number(a_number):
    """Return True if input is numerical. A helper function."""
    try:
        result = float(a_number)
        if np.isnan(result):
            return False
        return True
    except (ValueError, TypeError, NameError):
        return False


def do_fill(an_array):
    """Return a new numpy array with missing values replaced with medians.

    an_array: a one-dimensional numpy array.
    This is a helper function called by the function fill_medians().
    """
    new_array = np.copy(an_array)
    FlagGood = [is_number(element) for element in new_array]
    FlagBad = [not is_number(element) for element in new_array]
    if new_array[FlagBad].size > 0:  # If there are missing values
        new_array[FlagBad] = np.median(new_array[FlagGood].astype(float))

    return new_array


def fill_medians(array_like):
    """Return a new numpy array with missing values replaced with medians.

    array_like: a one or multi-dimensional array-like object.
    Missing values are replaced column-wise in case of a multi-dim.
    array.
    """
    # Make a copy of the input array and if the input is a usual Python
    # list convert it into a numpy array
    new_array = np.copy(array_like)

    # If clauses for one-dim. arrays, else clause for multi-dim. arrays
    # Eg. I treat [[1, 2, 3]] and [1, 2, 3] as one-dim. here
    if new_array.shape[0] == 1:
        new_array = do_fill(new_array[0])
    elif len(new_array.shape) == 1:
        new_array = do_fill(new_array)
    # Eg. [[ 1,  2], [3, 4]]
    else:
        # Transpose the array to operate on columns like on rows
        new_array = new_array.T
        for i, column in enumerate(new_array):
            new_array[i, :] = do_fill(column)
        # Transpose back to original form
        new_array = new_array.T

    new_array = new_array.astype(float)

    return new_array


# 6.  Comments explaining the code blocks: INSIDE FUNCTIOMNS
# 7.  Summary comment block on how your dataset has been cleaned up: BELOW


if __name__ == "__main__":

    # The first one-dimentional array contains a list of people ages
    # The function remove_outliers() removes outliers from this list where
    # outliers are defined as elements which are more than 2 sd
    # away from the mean
    ages_data = get_ages()
    cleaned_ages = remove_outliers(ages_data)
    print("Array 1 (with outliers):", "\n", ages_data)
    print("Modified array 1 (outliers removed):", "\n", cleaned_ages)
    # Check that my remove function produces expected outcome on simple
    # input
    test1 = [2, 1, 1, 99, 1, 5, 3, 1, 4, 3]
    result1 = remove_outliers(test1)
    expected1 = [2, 1, 1, 1, 5, 3, 1, 4, 3]
    np.testing.assert_array_equal(result1, expected1)

    # The second array, which is now multi-dimensional, contains
    # some movie data. The function replace_outliers_median()
    # replaces outliers (calculated by columns) with medians of
    # respective columns
    movies = get_movie_data()
    cleaned_movies = replace_outliers_median(movies)
    print("\n")
    print("Array 2 (with outliers):", "\n", movies)
    print("Modified array 2 (outliers replaced with medians):",
          "\n", cleaned_movies
          )
    # Check that my replace function produces expected outcome on simple
    # input
    test2 = [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [6, 11], [99, 99]]
    result2 = replace_outliers_median(test2)
    expected2 = [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [6, 11], [4, 9]]
    np.testing.assert_array_equal(result2, expected2)

    # The third 13x4 array contains some country data with missing values.
    # The function fill_medians() replaces missing values
    # with medians calculated by columns (except when the array is
    # one-dimensional).
    country = get_usa_data()
    cleaned_country = fill_medians(country)
    print("\n")
    print("Array 3 (with missing values):",
          "\n",
          country
          )
    print("Modified array 3 (missing values replaced with medians):",
          "\n",
          cleaned_country
          )

    # Check that the function is_number() produces expected outcome
    assert is_number(None) is False
    assert is_number("NAN") is False
    assert is_number(float("nan")) is False
    assert is_number("0.1") is True

    # Check that my 'fill-in' functions produce expected outcome
    # First, column-wise
    test3 = [[1, 3, 6], [2, 4, 7], [None, 5, None], [3, 6, None]]
    result3 = fill_medians(test3)
    expected3 = [[1, 3, 6], [2, 4, 7], [2, 5, 6.5], [3, 6, 6.5]]
    np.testing.assert_array_equal(result3, expected3)

    # Now, row-wise, for one-dim. arrays only
    test4 = [1, 2, 3, 4, 5, 6, 7, None, 8, 9, 10]
    result4 = fill_medians(test4)
    expected4 = [1, 2, 3, 4, 5, 6, 7, 5.5, 8, 9, 10]
    np.testing.assert_array_equal(result4, expected4)

    test5 = [[1, 2, 3, 4, 5, 6, 7, None, 8, 9, 10]]
    result5 = fill_medians(test5)
    expected5 = [1, 2, 3, 4, 5, 6, 7, 5.5, 8, 9, 10]
    np.testing.assert_array_equal(result5, expected5)
