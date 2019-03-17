#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains Lesson 8 Assignment.

Classification (by using DecisionTreeClassifier or LogisticRegression).

The data set is Census Income Data Set.
Ultimate purpose: Predict whether income exceeds $50K/yr based on census data.
The data set has 14 attributes and 1 class label - 'income'.
https://archive.ics.uci.edu/ml/datasets/Census+Income
http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
"""
import os
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import LogisticRegression

# This allows to print to console all categories
pd.set_option("max_column", 18)


def print_hist(a_column, a_category):
    """Print a histogram for a given column (series) and category (str)."""
    plt.hist(a_column)
    plt.title("Histogram for {} variable".format(a_category))
    plt.xlabel(a_category)
    plt.ylabel("frequency")
    plt.show()


def print_summ_stats(category, a_column, limit_hi, limit_lo, num_outliers):
    """Print summary stats for a given column (a pandas series object)."""
    print("""\nThe '{}' category:
        Count: {}
        Distinct: {}
        Min_value: {}
        Max_value: {}
        Median: {}
        Mean: {:.3f}
        St. dev.: {:.3f}
        Limit_Low: {:.3f}
        Limit_High: {:.3f}
        # outliers: {:.3f}
        """
          .format(category,
                  a_column.count(),
                  len(a_column.unique()),
                  np.min(a_column),
                  np.max(a_column),
                  np.median(a_column),
                  np.mean(a_column),
                  np.std(a_column),
                  limit_lo,
                  limit_hi,
                  num_outliers
                  )
          )


def get_outliers(a_dataframe, show=True):
    """Return an outliers list, call print_hist() and print_summ_stats().

    Returned value: a list with pandas.core.series.Series with booleans.
    """
    outliers_list = []
    for category in a_dataframe.dtypes.keys():
        try:
            column = a_dataframe.loc[:, category]
            mean = np.mean(column)  # check if category is numeric
        except TypeError:
            pass
        else:
            # print_hist(column, category)
            st_dev = np.std(column)
            limit_hi = mean + 2 * st_dev
            limit_lo = mean - 2 * st_dev
            flag_bad = (column < limit_lo) | (column > limit_hi)
            if category != "fnlwgt":  # skip 'fnlwgt' var. 'cos I'll delete it
                outliers_list.append(flag_bad)
            if show:
                num_outliers = sum(flag_bad)
                print_summ_stats(category, column,
                                 limit_hi, limit_lo,
                                 num_outliers
                                 )
    return outliers_list


def count_missing(a_dataframe):
    """Return a list of tuples (variablle_name, # of ' ?') for the DFrame.

    Count ? marks in non-numerical columns.
    """
    missing_count = []
    for category in a_dataframe.dtypes.keys():
        try:
            np.mean(a_dataframe.loc[:, category])  # skipping numeric
        except TypeError:
            count = sum(a_dataframe.loc[:, category] == " ?")
            if count > 0:
                missing_count.append((category, count))

    return missing_count


def load_data(a_url, path, column_headers):
    """Return a DataFrame containing a dataset.

    Load data from a_url or local path, attach column_headers to data,
    and save csv-file to disk if it's not already there.
    """
    print("Trying to load the data set from a local file if it exists")
    try:
        data_set = pd.read_csv(path)
    except FileNotFoundError:
        print("File not found, loading from Internet and saving to disk...")
        data_set = pd.read_csv(a_url, header=None)
        data_set.columns = column_headers
        data_set.to_csv(path, sep=",", index=False)

    return data_set


def split80_20(a_dataset, label, show=True):
    """Split data set ~80/20 into training/test sets.

    Basic approach to splitting is from stackoverflow.
    """
    np.random.seed(0)
    msk = np.random.rand(len(a_dataset)) < 0.8
    training_data = a_dataset[msk]
    test_data = a_dataset[~msk]
    if show:
        print("training dataset size:", training_data.shape)
        print("test dataset size:", test_data.shape)

    # Split training set into data X and class labels Y
    training_target = training_data.loc[:, label].values.astype(str)
    training_data = training_data.drop(label, axis=1)

    # Split test set into data XX and class labels YY
    test_target = test_data.loc[:, label].values.astype(str)
    test_data = test_data.drop(label, axis=1)

    return training_data, test_data, training_target, test_target


def print_intro():
    """Print a short narrative on the data preparation."""
    print("""\nDATA PREPARATION

    This will include:  - removing outliers
                        - dealing with missing value
                          (a) imputation (most frequent value), and
                          (b) removal
                        - consolidation of categories within variables
                        - removal of a very rare category within a variable
                        - removal of extraneous columns
                        - binning numerical variables
                        - (no z-normalization is used by default on any of
                           the variables (but can be applied via 'znormal'
                           switch) - it produces slightly worse results for
                           both DecisionTreeClassifier and
                           LogisticRegression)
                        - one-hot encoding of the entire data set (except
                          class labels)
                        - replacing the two class labels with "0" and "1"
                          (i.e. income '<=50K' and '>50K'; for ease of
                          computations only, not required by the classifier)
    """)


def remove_outliers(a_dataset):
    """Return the dataset with removed outliers."""
    print("""\n - REMOVING OUTLIERS
    """)
    dataset_copy = deepcopy(a_dataset)
    # Get outliers for numerical categories
    all_outliers_list = get_outliers(a_dataset, show=False)
    # Take a union of all outlier columns
    new_out_df = pd.DataFrame(all_outliers_list).T
    new_out_df.loc[:, "union"] = (new_out_df["age"]
                                  | new_out_df["education-years"]
                                  | new_out_df["capital-gain"]
                                  | new_out_df["capital-loss"]
                                  | new_out_df["hours-per-week"]
                                  )
    # Get a dataset without outliers
    return_dataset = dataset_copy.loc[~new_out_df["union"], :]

    print(sum(new_out_df["union"]), "rows were removed")

    return return_dataset


def impute_missing(a_dataset):
    """Return the dataset with imputed values."""
    dataset_copy = deepcopy(a_dataset)
    print("""\n\n - IMPUTING MISSING VALUES""")

    print("\nNumber of missing values (in non-numerical variables only):\n")
    print(count_missing(dataset_copy))

    missing_values = dataset_copy.loc[:, "native-country"] == " ?"
    print("""\n\t\tThe column 'native-country' has {} missing values
          denoted by ' ?'. I will replace ' ?' with the most frequent
          value, 'United-States'.
          """.format(sum(missing_values)
                     )
          )
    dataset_copy.loc[missing_values, "native-country"] = " United-States"

    missing_values = dataset_copy.loc[:, "workclass"] == " ?"
    print("""\n\t\tThe column 'workclass' has {} missing values
          denoted by '?'. I will replace '?' with the most frequent
          value, 'Private'
          """.format(sum(missing_values)
                     )
          )
    dataset_copy.loc[missing_values, "workclass"] = " Private"

    return dataset_copy


def remove_missing(a_dataset):
    """Return the dataset with removed missing values."""
    dataset_copy = deepcopy(a_dataset)
    print("""\n\n - REMOVING MISSING VALUES

          \tI will remove rows with missing values in the "occupation"
          category.
          """)

    question_marks = dataset_copy.loc[:, "occupation"] == " ?"
    print("\tI will remove", sum(question_marks), "values")
    dataset_copy = dataset_copy.loc[~question_marks, :]

    return dataset_copy


def consolidate_categories(a_dataset):
    """Return the dataset with consolidated categories."""
    dataset_copy = deepcopy(a_dataset)
    print("""\n\n - CONSOLIDATING CATEGORIES

      \tI will consolidate categories in the 'native-country'
      variable into 2 categories, 'US' and 'non-US', because the vast
      majority of cases are 'US' and all other countries account for
      a very small amount of cases.
      """)

    non_united_states = dataset_copy.loc[:, "native-country"] != " United-States"
    # Abbreviate ' United-States' as 'US'
    dataset_copy.loc[~non_united_states, "native-country"] = "US"
    dataset_copy.loc[non_united_states, "native-country"] = "Non-US"
    print(dataset_copy.loc[:, "native-country"].value_counts())

    print("""\n\t\tI will consolidate categories in 'workclass':
          """)
    replace_list = [" State-gov", " Federal-gov", " Local-gov"]
    print("{} --> 'gov'".format(replace_list))
    dataset_copy.loc[:, "workclass"].replace(replace_list, "gov", inplace=True)

    replace_list = [" Private", " Self-emp-not-inc", " Self-emp-inc"]
    print("{} --> ''private'\n".format(replace_list))
    dataset_copy.loc[:, "workclass"].replace(replace_list, "private", inplace=True)
    print(dataset_copy.loc[:, "workclass"].value_counts())

    print("""
          ' Without-pay' will go to the most frequent category 'private'
          and I will remove 'Never-worked' cases
          """)
    dataset_copy.loc[:, "workclass"].replace(" Without-pay",
                                             "private",
                                             inplace=True)

    never_worked_values = dataset_copy.loc[:, "workclass"] == ' Never-worked'
    dataset_copy = dataset_copy.loc[~never_worked_values, :]
    print(dataset_copy.loc[:, "workclass"].value_counts())

    print("""\n\t\tI will consolidate categories in 'marital-status':
          """)

    replace_list = [' Never-married', ' Separated',
                    ' Widowed', ' Divorced', ' Married-spouse-absent']

    print("{} --> 'unmarried'".format(replace_list))
    dataset_copy.loc[:, "marital-status"].replace(replace_list,
                                                  "unmarried",
                                                  inplace=True)

    replace_list = [' Married-civ-spouse', ' Married-AF-spouse']
    print("{} --> 'married'\n".format(replace_list))
    dataset_copy.loc[:, "marital-status"].replace(replace_list,
                                                  "married",
                                                  inplace=True)

    print(dataset_copy.loc[:, "marital-status"].value_counts())

    print("""\n\t\tI will consolidate categories in 'occupation':
          """)

    replace_list = [' Handlers-cleaners', ' Craft-repair',
                    ' Transport-moving', ' Farming-fishing',
                    ' Priv-house-serv']

    print("{} --> 'Manual'\n".format(replace_list))
    dataset_copy.loc[:, "occupation"].replace(replace_list,
                                              "Manual",
                                              inplace=True)

    return dataset_copy


def remove_columns(a_dataset):
    """Return the dataset with removed columns."""
    dataset_copy = deepcopy(a_dataset)

    print("""\n\n - REMOVING COLUMNS

      \tI want to remove the 'fnlwgt' column since
      it relates to some obscure calculated statistic for
      similarity of people's demographic characteristics within each state.

      \tI also remove the column 'education' because it has a numerical
      equivalent 'education-years'.
      """)

    dataset_copy.drop(columns=["fnlwgt", "education"], inplace=True)

    return dataset_copy


def apply_binning(a_dataset, vars_to_bin):
    """Return the dataset with binned variables."""
    dataset_copy = deepcopy(a_dataset)
    print("""\n\n - BINNING

      I'll use equal-width binning with 3 bins for the variables {}
      """
          .format(vars_to_bin)
          )

    for cat in vars_to_bin:
        dataset_copy.loc[:, cat] = pd.cut(dataset_copy.loc[:, cat],
                                          bins=3,
                                          labels=['L', 'M', 'H']
                                          )
    return dataset_copy


def apply_znorm(a_dataset, vars_to_znorm):
    """Return the dataset with z-normalized variables."""
    dset = deepcopy(a_dataset)
    print("""\n\n - Z-NORMALIZING

      I'll z-normalize the variables {}
      """
          .format(vars_to_znorm)
          )

    for cat in vars_to_znorm:
        dset[cat] = preprocessing.StandardScaler().fit_transform(dset[[cat]])

    return dset


def one_hot_encode(a_dataset, a_prefix_dict):
    """Return the dataset with one-hot-encoded variables."""
    dataset_copy = deepcopy(a_dataset)
    print("""\n\n - ONE-HOT ENCODING""")

    # Extract column headers from the above dict
    clmn_headers = list(a_prefix_dict.keys())

    # One-hot encode the variables listed in clmn_headers
    dataset_copy = pd.get_dummies(dataset_copy,
                                  columns=clmn_headers,
                                  prefix=a_prefix_dict
                                  )
    print("\nResulting dataset shape:", dataset_copy.shape)

    return dataset_copy


if __name__ == "__main__":
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    source_data_file = "adult.csv"

    columns_headers = ["age", "workclass", "fnlwgt", "education",
                       "education-years", "marital-status", "occupation",
                       "relationship", "race", "sex", "capital-gain",
                       "capital-loss", "hours-per-week",
                       "native-country", "income"]

    adult = load_data(url, source_data_file, columns_headers)

    print("\nThe shape of the original Adult Census Income data frame:")
    print(adult.shape)

    # Change to True to apply LogisticRegression
    logit = False
    # This switch allows to z-normalize several variables instead of binning
    znormal = False

    print("""\nPURPOSE: Predict if income exceeds $50K/yr based on census data

    By default, I use DecisionTreeClassifier, but you can also apply
    LogisticRegression by changing the value of 'logit' to True.
    """)

    print_intro()

    adult = remove_outliers(adult)

    adult = impute_missing(adult)

    adult = remove_missing(adult)

    adult = consolidate_categories(adult)

    adult = remove_columns(adult)

    # Variables to be binned
    binning_vars = ["capital-gain", "capital-loss"]
    # Variables to be z-normalized if 'znormal' is True, else to be binned
    znorm_vars = ["age", "hours-per-week", "education-years"]
    # Abbreviated prefixes for dummy variables for one-hot encoding
    prefix_dict = {"workclass": "wrk_cla",
                   "marital-status": "marital",
                   "sex": "sex",
                   "capital-gain": "cap_gain",
                   "occupation": "occu",
                   "relationship": "rela",
                   "race": "race",
                   "capital-loss": "cap_loss",
                   "native-country": "nat_cou"
                   }
    # Exclude z-normalized variables, if any, from one-hot encoding
    add_columns = {"age": "age",
                   "hours-per-week": "hr_wk",
                   "education-years": "ed_yrs"
                   }

    # By default, z-normalization is not used; binning is used
    if znormal:
        adult = apply_znorm(adult, znorm_vars)
    else:
        binning_vars += znorm_vars  # Extend the list of variables to bin
        prefix_dict.update(add_columns)  # Extend the dict to one-hot encode

    adult = apply_binning(adult, binning_vars)

    adult_encoded = one_hot_encode(adult, prefix_dict)

    print("Replacing class labels ' <=50K' and ' >50K' as '0' and '1'")
    adult_encoded.loc[:, "income"].replace(' <=50K', "0", inplace=True)
    adult_encoded.loc[:, "income"].replace(' >50K', "1", inplace=True)

    print("""\n\nCLASSIFICATION (using {})

    For this assignment, I use sklearn's DecisionTreeClassifier

    You can switch to LogisticRegression by setting 'logit' to True
    (but the results are even worse)
    """
          .format("LogisticRegression" if logit is True else "DecisionTreeClf")
          )

    print("\n\tSplit the dataset ~80/20 into train/test/labelTrain/labelTest:")
    X, XX, Y, YY = split80_20(adult_encoded, label='income')

    print("""\n\tInstantiate a classifier class object and use its fit
    and predict methods on the training set""")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    # Use its predict method to guess labels for the test set
    predicted_YY = clf.predict(XX.values)

    # Logistic regression classifier
    if logit:
        C_parameter = 50. / len(X)
        clf = LogisticRegression(C=C_parameter,
                                 multi_class='ovr',
                                 penalty='l1',
                                 solver='saga',
                                 tol=0.1)
        clf.fit(X, Y)
        predicted_YY = clf.predict(XX.values)

    # Tell something about the model's performance on this data set
    print("\n\tCorrectly predicted by model (correct/total): {:.2f}%"
          .format(
                  (sum(predicted_YY == YY) / len(YY)) * 100
                  )
          )
    # The following calculates % of low income persons in the test set
    print("""\tIn how many cases would I be right if I just rated everyone as
          low income in the test set? The answer is {:.2f}%"""
          .format(
                  (1 - sum(YY.astype(int)) / len(YY)) * 100
                  )
          )
    print("""\n\tTherefore, my conclusion is that my model is not
          good at all. Probably, I should reduce the number of variables
          somehow, but I haven't figured out how yet.
          """)

    print("""\n\tCreate a dataframe of predicted and actual values
    and print a two-way table by using panda's crosstab function\n""")
    predicted_YY = predicted_YY.astype(int)
    YY = YY.astype(int)
    labels_dict = {'Predicted': predicted_YY, 'Actual': YY}

    # Create a pandas dataframe from a dict with numpy ndarrays
    predicted_vs_actuals = pd.DataFrame(labels_dict)
    # Make a two-way table of actuals and predicted
    two_way_tab = pd.crosstab(index=predicted_vs_actuals['Predicted'],
                              columns=predicted_vs_actuals['Actual'],
                              margins=True,
                              margins_name='Totals'
                              )
    print(two_way_tab)
    true_pos = two_way_tab[1][1]
    false_neg = two_way_tab[1][0]
    true_neg = two_way_tab[0][0]
    false_pos = two_way_tab[0][1]
    recall = true_pos / (true_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    print("\n\tCorrectly labeled low income (0's in the table above): {:.2f}%"
          .format(specificity * 100)
          )
    print("\tCorrectly labeled high income (1's in the table above): {:.2f}%"
          .format(recall * 100)
          )

    # This runs for DecisionTreeClf and is switched off for LogisticRegr.
    if not logit:
        print("""\n\tRange features by importance according to
        the classifier. Show the top 5 features:\n""")
        # Combine one-hot encoded column names and their importance into a dict
        d = dict(zip(adult_encoded.columns, clf.feature_importances_))
        # Print the dict sorted by values
        print(sorted(d.items(), key=lambda x: x[1], reverse=True)[:5])

    print("""\n\nSAVING TRUE AND PREDICTED LABELS TO AN CSV-FILE ON DISK""")
    predicted_vs_actuals.to_csv("labels.csv", sep=",", index=False)
    if "labels.csv" in os.listdir():
        print("\nFile {} saved in {}".format("labels.csv", os.getcwd()))
