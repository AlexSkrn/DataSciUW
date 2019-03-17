#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This files contains Lesson 8 Assignment.
Classification.

The data set is Census Income Data Set.
Ultimate purpose: Predict whether income exceeds $50K/yr based on census data.
The data set has 14 attributes and 1 class label - 'income'.
https://archive.ics.uci.edu/ml/datasets/Census+Income
http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
"""
#import os
import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import preprocessing
#from sklearn.cluster import KMeans
from scipy import stats
from sklearn import tree
#import graphviz
from copy import deepcopy

# This allows to print to console all categories
pd.set_option("max_column", 18)


def print_hist(a_column, a_category):
    """Print a histogram for a given column (series) and category (str)."""
    plt.hist(a_column)
    plt.title("Histogram for {} variable".format(a_category))
    plt.xlabel(a_category)
    plt.ylabel("frequency")
    plt.show()


def print_stats(category, a_column, limit_hi, limit_lo, num_outliers):
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
    """Return an outliers list, call print_hist() and print_stats().

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
                print_stats(category, column,
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
    print("Try loading the data set from a local file if it exists.")
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


if __name__ == "__main__":
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    source_data_file = "adult.csv"

    columns_headers = ["age", "workclass", "fnlwgt", "education",
                       "education-years", "marital-status", "occupation",
                       "relationship", "race", "sex", "capital-gain",
                       "capital-loss", "hours-per-week",
                       "native-country", "income"]

    target_data_file = "Dataset.csv"

    adult = load_data(url, source_data_file, columns_headers)

    print("\nThe shape of the Adult Census Income data frame:")
    print(adult.shape)

#    print("\nThe first few rows of the date frame:")
#    print(adult.head())
#
#    print("\nThe data types of the columns in the data frame:")
#    print(adult.dtypes)

    print("\nGet summary statistics and outliers for numerical categories")
    all_outliers_list = get_outliers(adult, show=False)

#    print("\nLet's plot all the numeric columns against each other.")
#    scatter_matrix(adult, figsize=[8, 8], s=50)
#    plt.title("Scatter Plot of All Numerical Variables")
#    plt.show()

    print("\nNumber of missing values (in non-numerical variables only):\n")
    print(count_missing(adult))

#    print("""\n\nREMOVING OUTLIERS.
#
#    \tIf I wanted to remove outliers, how many rows would I remove?
#
#    \tTo answer that question, I'll create a new dataframe for outliers
#    (except for 'fnlwgt' variable which I will delete later)
#    with a new calculated column which is the union of the previous
#    columns. This will give me a list of outliers for the entire dataset.\n
#    """)

    new_out_df = pd.DataFrame(all_outliers_list).T
    new_out_df.loc[:, "union"] = (new_out_df["age"]
                                  | new_out_df["education-years"]
                                  | new_out_df["capital-gain"]
                                  | new_out_df["capital-loss"]
                                  | new_out_df["hours-per-week"]
                                  )

#    print(new_out_df.head())
#    print("\n")
    print(sum(new_out_df["union"]), "rows to remove out of", adult.shape[0])

#    print("\nWhat is the distribution of counts in the 'income' column?")
#    print(pd.crosstab(adult["income"], columns="count"))
#
#    print("\nHow does this count look like after removal of the outliers?")
    adult = adult.loc[~new_out_df["union"], :]
#    print(pd.crosstab(adult["income"], columns="count"))

#    print("""\t\tThe decrease looks roughly proportional in both categories.
#          The resulting shape of the dataframe, without rows with outliers:
#          """)
#    print(adult.shape)

#    print("""\n\nZ-NORMALIZATION
#
#    I will z-normalize the 'age' variable using StandardScaler\n
#    """)
#    age_df = adult[['age']]
#    standard_scaler = preprocessing.StandardScaler().fit(age_df)
#    adult["age_z"] = standard_scaler.transform(age_df)
#    print("z-normalized 'age_z' column's mean and std are now {:.3f} and {:.3f}"
#          .format(adult.age_z.mean(),
#                  adult.age_z.std()
#                  )
#          )
#
#    print("""
#
#    I will z-normalize the 'hours-per-week' variable using StandardScaler\n
#    """)
#    hr_wk_df = adult[['hours-per-week']]
#    standard_scaler = preprocessing.StandardScaler().fit(hr_wk_df)
#    adult["hours-per-week_z"] = standard_scaler.transform(hr_wk_df)
#    print("""z-normalized 'hours-per-week_z' column's mean and std are now {:.3f}
#    and {:.3f}"""
#          .format(adult.loc[:, "hours-per-week_z"].mean(),
#                  adult.loc[:, "hours-per-week_z"].std()
#                  )
#          )

#    plt.hist(adult.loc[:, "age"])
#    plt.title("Age Category after Z-normalization")
#    plt.xlabel("normalized age")
#    plt.ylabel("frequency")
#    plt.show()

    print("""\n\nEQUIVALENT VARIABLES

    \tThe dataset contains two equivalent variables,
    'education' and 'education-years'.

    \tIt is easy to see which 'education' value corresponds to which
    'education-years' value by constructing a two-way table of counts
    as follows. For example, 1 is 'Preschool' and 16 is 'Doctorate'.
    I will remove 'education' and will bin 'education-years' later.
    """)

    print(pd.crosstab(index=adult["education-years"].astype("str"),
                      columns=adult["education"]
                      )
          )


#        adult[cat].value_counts().plot(kind="bar", title=cat)
#        plt.ylabel("Count")
#        plt.xlabel("Binned {}".format(cat))
#        plt.show()
#        print(adult[cat].value_counts(), "\n")

    print("""\n\nIMPUTING VALUES""")

    missing_values = adult.loc[:, "native-country"] == " ?"
    print("""\n\t\tThe column 'native-country' has {} missing values
          denoted by ' ?'. I will replace ' ?' with the most frequent
          value, 'United-States'.
          """.format(sum(missing_values)
                     )
          )
    adult.loc[missing_values, "native-country"] = " United-States"

    missing_values = adult.loc[:, "workclass"] == " ?"
    print("""\n\t\tThe column 'workclass' has {} missing values
          denoted by '?'. I will replace '?' with the most frequent
          value, 'Private'
          """.format(sum(missing_values)
                     )
          )
    adult.loc[missing_values, "workclass"] = " Private"

    print("""\n\nCONSOLIDATING CATEGORIES

          \tI want to consolidate categories in the 'native-country'
          variable into 2 categories, 'US' and 'non-US', because the vast
          majority of cases are 'US' and all other countries account for
          a very small amount of cases.
          """)

    non_united_states = adult.loc[:, "native-country"] != " United-States"
    # Abbreviate ' United-States' as 'US'
    adult.loc[~non_united_states, "native-country"] = "US"
    adult.loc[non_united_states, "native-country"] = "Non-US"
    print(adult.loc[:, "native-country"].value_counts())

    print("""\n\t\tI will consolidate categories in 'workclass':
          """)
    replace_list = [" State-gov", " Federal-gov", " Local-gov"]
    print("{} --> 'gov'".format(replace_list))
    adult.loc[:, "workclass"].replace(replace_list, "gov", inplace=True)

    replace_list = [" Private", " Self-emp-not-inc", " Self-emp-inc"]
    print("{} --> ''private'\n".format(replace_list))
    adult.loc[:, "workclass"].replace(replace_list, "private", inplace=True)
    print(adult.loc[:, "workclass"].value_counts())

    print("""
          ' Without-pay' will go to the most frequent category 'private'
          and I will remove 'Never-worked' cases
          """)
    adult.loc[:, "workclass"].replace(" Without-pay",
                                        "private",
                                        inplace=True)

    never_worked_values = adult.loc[:, "workclass"] == ' Never-worked'
    adult = adult.loc[~never_worked_values, :]
    print(adult.loc[:, "workclass"].value_counts())

    print("""\n\t\tI will consolidate categories in 'marital-status':
          """)

    replace_list = [' Never-married', ' Separated',
                    ' Widowed', ' Divorced', ' Married-spouse-absent']

    print("{} --> 'unmarried'".format(replace_list))
    adult.loc[:, "marital-status"].replace(replace_list,
                                             "unmarried",
                                             inplace=True)

    replace_list = [' Married-civ-spouse', ' Married-AF-spouse']
    print("{} --> 'married'\n".format(replace_list))
    adult.loc[:, "marital-status"].replace(replace_list,
                                             "married",
                                             inplace=True)

    print(adult.loc[:, "marital-status"].value_counts())

    print("""\n\t\tI will consolidate categories in 'occupation':
          """)

    replace_list = [' Handlers-cleaners', ' Craft-repair',
                    ' Transport-moving', ' Farming-fishing',
                    ' Priv-house-serv']

    print("{} --> 'Manual'\n".format(replace_list))
    adult.loc[:, "occupation"].replace(replace_list,
                                         "Manual",
                                         inplace=True)

    print(adult.loc[:, "occupation"].value_counts())

    print("""\n\nREMOVING MISSING VALUES

          \tI already replaced missing values in the 'workclass'
          and 'native-country' categories. Now I will remove rows
          with missing values in the "occupation" category.
          """)

    question_marks = adult.loc[:, "occupation"] == " ?"
    print("I will remove", sum(question_marks), "values. The result is:")
    adult = adult.loc[~question_marks, :]
    print(adult.loc[:, "occupation"].value_counts())

    print("""\n\nREMOVING COLUMNS

          \tI want to remove the 'fnlwgt' column since
          it relates to some obscure calculated statistic for
          similarity of people's demographic characteristics within each state.

          \tI also remove the column 'education' because it has a numerical
          equivalent 'education-years'.
          """)

    adult = adult.drop(columns="fnlwgt")
    adult = adult.drop(columns="education")

#    print("""\n\nONE-HOT ENCODING""")
#    print("""\n\t\tI will one-hot encode the 'relationship' variable which
#          has 6 categories.
#          """)
#    adult = pd.get_dummies(adult, columns=["relationship"], prefix="relat")

#    print("""\n\nSAVING TO DISK""")
#    adult.to_csv(target_data_file, sep=",", index=False)
#    if target_data_file in os.listdir():
#        print("File {} saved in {}".format(target_data_file, os.getcwd()))



#    print(""""
#    NEW CODE FOR LESSON 7 ASSIGNMENT.
#    """)
#    print("""Trying to make a scatter plot of age and hour-per-week
#          after k-means analysis with 4 clusers.
#
#          Actually, it doesn't seem to matter how many clusters I
#          choose because the dots fill in the entire plot space.
#          """)
#    age_hours = adult[['age', 'hours-per-week']].copy()
#    age_hours_std = stats.zscore(age_hours[['age', 'hours-per-week']])


    print("""\n\nBINNING before creating dummies""")

    binning_vars = ["hours-per-week", "capital-gain",
                    "capital-loss", "education-years", "age"]

    print("\nI'll use equal-width binning with 3 bins for the variables {}\n"
          .format(binning_vars)
          )

    for cat in binning_vars:
        adult.loc[:, cat] = pd.cut(adult.loc[:, cat],
                                     bins=3,
                                     labels=['L', 'M', 'H']
                                     )

    print("""ONE-HOT ENCODE AND CLASSIFY""")

#    adult_encoded = adult.copy()
    adult_encoded = deepcopy(adult)

    # Create dummies for data attributes
    prefix_dict = {"workclass": "wrk_cla",
                   "marital-status": "marital",
                   "sex": "sex",
                   "capital-gain": "cap_gain",
                   "age": "age",
                   "education-years": "ed_yrs",
                   "occupation": "occu",
                   "relationship": "rela",
                   "race": "race",
                   "capital-loss": "cap_loss",
                   "hours-per-week": "hr_wk",
                   "native-country": "nat_cou"
                   }
    clmn_headers = list(prefix_dict.keys())

    scores = []
    # Drop one column in the dataset at a time
    for col_head in clmn_headers:
        print("\nLeaving out", col_head)
        temp_clmn_list = list(prefix_dict.keys())
        temp_prefix_d = deepcopy(prefix_dict)
        # Drop one column
        try:
            temp_set = adult_encoded.drop(col_head, axis=1)
        except KeyError:
            print("Column head", col_head, "not found in the dataset")
        # Remove respective column from my list of headers
        try:
            temp_clmn_list.remove(col_head)
        except ValueError:
            print("Value", col_head, "not found in temp_clmn_list")
        # Remove respective column from my dict of prefixes for one-hot
        try:
            del temp_prefix_d[col_head]
        except KeyError:
            print("Key", col_head, "not found in temp_prefix_d")
        # One-hot encode the resulting dataset (i.e. minus one column)
        temp_set = pd.get_dummies(temp_set,
                                  columns=temp_clmn_list,
                                  prefix=temp_prefix_d)
        # Replace class labels ' <=50K' and ' >50K' as "0" and "1"
        temp_set.loc[:, "income"].replace(' <=50K', "0", inplace=True)
        temp_set.loc[:, "income"].replace(' >50K', "1", inplace=True)

        # Split dataset into training/test/label-train/label-test
        X, XX, Y, YY = split80_20(temp_set, label="income")

        # Instantiate a classifer class object and use its fit method
        # on the training set
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, Y)
        # Use its predict method to guess labels for the test set
        predicted_YY = clf.predict(XX.values)
        print("% correctly predicted by model:",
          sum(predicted_YY == YY)/len(YY))
        print("% correct just by rating all as low income:",
              1 - sum(YY.astype(int))/len(YY))

        true_low, true_high, false_low, false_high = 0, 0, 0, 0
        for i, predicted_label in enumerate(predicted_YY):
            true_label = YY[i]
            if predicted_label == "0" and true_label == "0":
                true_low += 1
            elif predicted_label == "1" and true_label == "1":
                true_high += 1
            elif predicted_label == "1" and true_label == "0":
                false_low += 1
            elif predicted_label == "0" and true_label == "1":
                false_high += 1

        print("true_low", true_low)
        print("true_high", true_high)
        print("false_low", false_low)
        print("false_high", false_high)
        print("control sum", sum([true_low,
                                 true_high,
                                 false_low,
                                 false_high]))
        print("ttl high income in testset:", sum(YY.astype(bool)))
        print("Error in classifying low income:")
        print(false_low / (false_low
                                             + true_low))
        print("Error in classifying high income:")
        print(false_high / (false_high
                                              + true_high))
    ############################
        print("""Create a dataframe of predicted and actual values.""")
        predicted_YY = predicted_YY.astype(int)
        YY = YY.astype(int)
        labels_dict = {'Predicted': predicted_YY, 'Actual': YY}
        predicted_vs_actuals = pd.DataFrame(labels_dict)
        two_way_tab = pd.crosstab(index=predicted_vs_actuals['Predicted'],
                    columns=predicted_vs_actuals['Actual'],
                    margins=True,
                    margins_name='Totals')
        print(two_way_tab)
        recall = two_way_tab[1][1]/(two_way_tab[1][1] + two_way_tab[1][0])
        print("% correctly labeled high income: {}".format(recall))
        scores.append(recall)

    neg_predictor = list(prefix_dict.keys())[scores.index(max(scores))]
    pos_predictor = list(prefix_dict.keys())[scores.index(min(scores))]
    print('negative_predictor', neg_predictor)
    print('positive_predictor', pos_predictor)
    print()
    d = dict(zip(temp_set.columns, clf.feature_importances_))
    print(sorted(d.items(), key=lambda x:x[1], reverse=True))
####################
#    adult_encoded = pd.get_dummies(adult_encoded,
#                                   columns=clmn_headers,
#                                   prefix=prefix_dict
#                                   )
#    print("""Replace class labels ' <=50K' and ' >50K' as "0" and "1"
#          """)
#    adult_encoded.loc[:, "income"].replace(' <=50K', "0", inplace=True)
#    adult_encoded.loc[:, "income"].replace(' >50K', "1", inplace=True)
#
#
#    X, XX, Y, YY = split80_20(adult_encoded, label='income')
#
#############
#
#    print("""Use the classifier class fit() method.""")
#    clf = tree.DecisionTreeClassifier()
#    clf = clf.fit(X, Y)
#
##    # Get a vector for prediction from the test dataset - example
##    print(clf.predict([XX.values[0]]))
##    print(XX.values[0])
##    print(clf.predict([XX.values[0]])[0] == YY[0])
#    # Get all predicted targets for the test data
#    predicted_YY = clf.predict(XX.values)
#    print("% correctly predicted by model:",
#          sum(predicted_YY == YY)/len(YY))
#    print("% correct just by rating all as low income:",
#          1 - sum(YY.astype(int))/len(YY))
#
#    true_low, true_high, false_low, false_high = 0, 0, 0, 0
#    for i, predicted_label in enumerate(predicted_YY):
#        true_label = YY[i]
#        if predicted_label == "0" and true_label == "0":
#            true_low += 1
#        elif predicted_label == "1" and true_label == "1":
#            true_high += 1
#        elif predicted_label == "1" and true_label == "0":
#            false_low += 1
#        elif predicted_label == "0" and true_label == "1":
#            false_high += 1
#
#    print("true_low", true_low)
#    print("true_high", true_high)
#    print("false_low", false_low)
#    print("false_high", false_high)
#    print("control sum", sum([true_low,
#                             true_high,
#                             false_low,
#                             false_high]))
#    print("ttl high income in testset:", sum(YY.astype(bool)))
#    print("Error in classifying low income:")
#    print(false_low / (false_low
#                                         + true_low))
#    print("Error in classifying high income:")
#    print(false_high / (false_high
#                                          + true_high))
#############################
#    print("""Create a dataframe of predicted and actual values.""")
#    predicted_YY = predicted_YY.astype(int)
#    YY = YY.astype(int)
#    labels_dict = {'Predicted': predicted_YY, 'Actual': YY}
#    predicted_vs_actuals = pd.DataFrame(labels_dict)
#    two_way_tab = pd.crosstab(index=predicted_vs_actuals['Predicted'],
#                columns=predicted_vs_actuals['Actual'],
#                margins=True,
#                margins_name='Totals')
#    print(two_way_tab)
#    recall = two_way_tab[1][1]/(two_way_tab[1][1] + two_way_tab[1][0])
#    print("% correctly labeled high income: {}".format(recall))

#    print("""SAVE A VIZUALIZATION to PDF""")
#    ft_names = list(training_data.columns)
#    dot_data = tree.export_graphviz(clf, out_file=None,
#                         feature_names=ft_names,
#                         class_names=["0", "1"],
#                         filled=True, rounded=True,
#                         special_characters=True)
#    graph = graphviz.Source(dot_data)
#    graph.render('adult')
#
