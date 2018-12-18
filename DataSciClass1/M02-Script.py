#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:58:38 2018

@author: alexskrn

This files contains Milestone 2 Assignment.
Data Preparation.

The data set is Census Income Data Set.
Ultimate purpose: Predict whether income exceeds $50K/yr based on census data.
The data set has 14 attributes and 1 class label - 'income'.
https://archive.ics.uci.edu/ml/datasets/Census+Income
http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
"""
import os
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

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


def get_outliers(a_dataframe):
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


if __name__ == "__main__":
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    source_data_file = "adult.csv"

    columns_headers = ["age", "workclass", "fnlwgt", "education",
                       "education-years", "marital-status", "occupation",
                       "relationship", "race", "sex", "capital-gain",
                       "capital-loss", "hours-per-week",
                       "native-country", "income"]

    target_data_file = "M02-Dataset.csv"

    my_data = load_data(url, source_data_file, columns_headers)

    print("\nThe shape of the data frame:")
    print(my_data.shape)

    print("\nThe first few rows of the date frame:")
    print(my_data.head())

    print("\nThe data types of the columns in the data frame:")
    print(my_data.dtypes)

    print("\nGet summary statistics and outliers for numerical categories:")
    all_outliers_list = get_outliers(my_data)

    print("\nLet's plot all the numeric columns against each other.")
    scatter_matrix(my_data, figsize=[8, 8], s=50)
    plt.title("Scatter Plot of All Numerical Variables")
    plt.show()

    print("\nNumber of missing values in non-numerical variables:\n")
    print(count_missing(my_data))

    print("""\n\nREMOVING OUTLIERS.

    \tIf I wanted to remove outliers, how many rows would I remove?

    \tTo answer that question, I'll create a new dataframe for outliers
    (except for 'fnlwgt' variable which I will delete later)
    with a new calculated column which is the union of the previous
    columns. This will give me a list of outliers for the entire dataset.\n
    """)

    new_out_df = pd.DataFrame(all_outliers_list).T
    new_out_df.loc[:, "union"] = (new_out_df["age"]
                                  | new_out_df["education-years"]
                                  | new_out_df["capital-gain"]
                                  | new_out_df["capital-loss"]
                                  | new_out_df["hours-per-week"]
                                  )

    print(new_out_df.head())
    print("\n")
    print(sum(new_out_df["union"]), "rows to remove out of", my_data.shape[0])

    print("\nWhat is the distribution of counts in the 'income' column?")
    print(pd.crosstab(my_data["income"], columns="count"))

    print("\nHow does this count look like after removal of the outliers?")
    my_data = my_data.loc[~new_out_df["union"], :]
    print(pd.crosstab(my_data["income"], columns="count"))

    print("""\t\tThe decrease looks roughly proportional in both categories.
          The resulting shape of the dataframe, without rows with outliers:
          """)
    print(my_data.shape)

    print("""\n\nZ-NORMALIZATION

    I will z-normalize the 'age' variable using StandardScaler\n
    """)
    age_df = my_data[['age']]
    standard_scaler = preprocessing.StandardScaler().fit(age_df)
    my_data["age"] = standard_scaler.transform(age_df)
    print("z-normalized 'age' column's mean and std are now {:.3f} and {:.3f}"
          .format(my_data.age.mean(),
                  my_data.age.std()
                  )
          )

#    plt.hist(my_data.loc[:, "age"])
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

    print(pd.crosstab(index=my_data["education-years"].astype("str"),
                      columns=my_data["education"]
                      )
          )

    print("""\n\nBINNING""")

    binning_vars = ["hours-per-week", "capital-gain",
                    "capital-loss", "education-years"]

    print("\nI'll use equal-width binning with 3 bins for the variables {}\n"
          .format(binning_vars)
          )

    for cat in binning_vars:
        my_data.loc[:, cat] = pd.cut(my_data.loc[:, cat],
                                     bins=3,
                                     labels=['L', 'M', 'H']
                                     )
#        my_data[cat].value_counts().plot(kind="bar", title=cat)
#        plt.ylabel("Count")
#        plt.xlabel("Binned {}".format(cat))
#        plt.show()
        print(my_data[cat].value_counts(), "\n")

    print("""\n\nIMPUTING VALUES""")

    missing_values = my_data.loc[:, "native-country"] == " ?"
    print("""\n\t\tThe column 'native-country' has {} missing values
          denoted by ' ?'. I will replace ' ?' with the most frequent
          value, 'United-States'.
          """.format(sum(missing_values)
                     )
          )
    my_data.loc[missing_values, "native-country"] = " United-States"

    missing_values = my_data.loc[:, "workclass"] == " ?"
    print("""\n\t\tThe column 'workclass' has {} missing values
          denoted by '?'. I will replace '?' with the most frequent
          value, 'Private'
          """.format(sum(missing_values)
                     )
          )
    my_data.loc[missing_values, "workclass"] = " Private"

    print("""\n\nCONSOLIDATING CATEGORIES

          \tI want to consolidate categories in the 'native-country'
          variable into 2 categories, 'US' and 'non-US', because the vast
          majority of cases are 'US' and all other countries account for
          a very small amount of cases.
          """)

    non_united_states = my_data.loc[:, "native-country"] != " United-States"
    # Abbreviate ' United-States' as 'US'
    my_data.loc[~non_united_states, "native-country"] = "US"
    my_data.loc[non_united_states, "native-country"] = "Non-US"
    print(my_data.loc[:, "native-country"].value_counts())

    print("""\n\t\tI will consolidate categories in 'workclass':
          """)
    replace_list = [" State-gov", " Federal-gov", " Local-gov"]
    print("{} --> 'gov'".format(replace_list))
    my_data.loc[:, "workclass"].replace(replace_list, "gov", inplace=True)

    replace_list = [" Private", " Self-emp-not-inc", " Self-emp-inc"]
    print("{} --> ''private'\n".format(replace_list))
    my_data.loc[:, "workclass"].replace(replace_list, "private", inplace=True)
    print(my_data.loc[:, "workclass"].value_counts())

    print("""
          ' Without-pay' will go to the most frequent category 'private'
          and I will remove 'Never-worked' cases
          """)
    my_data.loc[:, "workclass"].replace(" Without-pay",
                                        "private",
                                        inplace=True)

    never_worked_values = my_data.loc[:, "workclass"] == ' Never-worked'
    my_data = my_data.loc[~never_worked_values, :]
    print(my_data.loc[:, "workclass"].value_counts())

    print("""\n\t\tI will consolidate categories in 'marital-status':
          """)

    replace_list = [' Never-married', ' Separated',
                    ' Widowed', ' Divorced', ' Married-spouse-absent']

    print("{} --> 'unmarried'".format(replace_list))
    my_data.loc[:, "marital-status"].replace(replace_list,
                                             "unmarried",
                                             inplace=True)

    replace_list = [' Married-civ-spouse', ' Married-AF-spouse']
    print("{} --> 'married'\n".format(replace_list))
    my_data.loc[:, "marital-status"].replace(replace_list,
                                             "married",
                                             inplace=True)

    print(my_data.loc[:, "marital-status"].value_counts())

    print("""\n\t\tI will consolidate categories in 'occupation':
          """)

    replace_list = [' Handlers-cleaners', ' Craft-repair',
                    ' Transport-moving', ' Farming-fishing',
                    ' Priv-house-serv']

    print("{} --> 'Manual'\n".format(replace_list))
    my_data.loc[:, "occupation"].replace(replace_list,
                                         "Manual",
                                         inplace=True)

    print(my_data.loc[:, "occupation"].value_counts())

    print("""\n\nREMOVING MISSING VALUES

          \tI already replaced missing values in the 'workclass'
          and 'native-country' categories. Now I will remove rows
          with missing values in the "occupation" category.
          """)

    question_marks = my_data.loc[:, "occupation"] == " ?"
    print("I will remove", sum(question_marks), "values. The result is:")
    my_data = my_data.loc[~question_marks, :]
    print(my_data.loc[:, "occupation"].value_counts())

    print("""\n\nREMOVING COLUMNS

          \tI want to remove the 'fnlwgt' column since
          it relates to some obscure calculated statistic for
          similarity of people's demographic characteristics within each state.

          \tI also remove the column 'education' because it has a numerical
          equivalent 'education-years'.
          """)

    my_data = my_data.drop(columns="fnlwgt")
    my_data = my_data.drop(columns="education")

    print("""\n\nONE-HOT ENCODING""")
    print("""\n\t\tI will one-hot encode the 'relationship' variable which
          has 6 categories.
          """)
    my_data = pd.get_dummies(my_data, columns=["relationship"], prefix="relat")

    print("""\n\nSAVING TO DISK""")
    my_data.to_csv(target_data_file, sep=",", index=False)
    if target_data_file in os.listdir():
        print("File {} saved in {}".format(target_data_file, os.getcwd()))

    print("""\n\nMY SUMMARY:
          The dataset had 6 numeric and 8 non-numerical variable, and 1
          class label - 'income'.

          (1)  I removed 5818 rows with outliers from 5 numeric variables
          (out of the total 32561 rows in the data set). I think such
          removal of outliers did not affect the prediction variable "income"
          because the removal affected both labels (i.e. '>50K' and '<=50K')
          proportionately.

          (2) I z-normalized the 'age' variable using the StandardScaler
          from the sklearn library. This was done as an exercise. I think
          I should probably bin this variable too. I am not yet sure
          how this will be used in any subsequent analysis.

          (3) I binned the following numerical variables:
              'hours-per-week',
              'capital-gain',
              'capital-loss', and
              'education-years',
          in each case using 3 bins.

          (4) I replaced missing values with the most frequent values:
              'native-country': 'United-States' used as the replacement value.
              'workclass': 'Private' used as the replacement value.

          (5) I consolidated categories in the following variables:
              'native-country' -> reduction of the number of categories to 2.
              'workclass' -> reduction of the number of categories to 2.
                              In the 'workclass', I also removed
                              7 'Never-worked' cases.
              'marital-status' -> reduction of the number of categories to 2.
              'occupation' -> reduction of the number of categories to 10
                              (from 14). In the 'occupation', I also removed
                              rows with missing values.

          (6) I removed the following columns:
              'fnlwgt': it contained some calculated data which is hard to
                         interpret.
              'education': it has a numerical equivalent - 'education-years'.

          (7) I one-hot encoded the 'relationship' variable with 6 categories.

          TO-DO:
          (8) I am not sure about how the data will be used in any later
              analysis. Probably, the 'occupation' and 'race'
              variables which have many categories should be converted
              into dummies. Probably, 'age' should be binned rather than
              z-normalized.
          """)

    print("""RESULTING DATASET DESCRIPTION:

            18 variables (incl. 6 dummies), no missing values, no outliers

            1) age: numerical, z-normalized
            2) workclass: 2 consolidated categories: 'gov', 'private'
            3) education-years: binned into 3 equal-width bins, L, M, H
            4) marital-status: 2 consolidated categories: married, unmarried
            5) occupation: 10 categories, with 'manual' being is a result
                         of consolidating 5 other categories
            6) race: categorial (5 categories), no changes vs. original dataset
            7) sex: categorial (2 categories), no changes vs. original dataset
            8) capital-gain: binned into 3 equal-width bins, L, M, H
            9) capital-loss: binned into 3 equal-width bins, L, M, H
            10) hours-per-week: binned into 3 equal-width bins, L, M, H
            11) native-country: 2 consolidated categories: 'US', 'Non-US'
            12) income: class label: <=50K, >50K

            The following are 6 dummy variables from the old 'relationship'
            variable with their approx. meanings. 'Relationship' means, in
            general, relationship to the head of household.

            13) relat_ Husband: a person married to and living
                                with a householder (in marriage)
            14) relat_ Not-in-family: a householder living alone or with
                                      non-relatives
            15) relat_ Other-relative: any household member related
                                       to the householder, but not included
                                       specifically in another relationship
                                       category
            16) relat_ Own-child: A never-married child under 18 years
                                  who is a son or daughter
                                  of the householder
            17) relat_ Unmarried: an adult who is unrelated
                                  to the householder, but shares living
                                  quarters and has a close personal
                                  relationship with the householder
            18) relat_ Wife: a person married to and living
                             with a householder (in marriage)
            """)
