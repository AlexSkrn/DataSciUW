#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:58:38 2018

@author: alexskrn

This files contains Lession 05 Assignment.
Categorical Data.

The data set is Census Income Data Set.
Ultimate purpose: Predict whether income exceeds $50K/yr based on census data.
The data set has 14 attributes.
https://archive.ics.uci.edu/ml/datasets/Census+Income
http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# This allows to print to console all categories
pd.set_option("max_column", 15)

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns_headers = ["age", "workclass", "fnlwgt", "education",
                   "education-years", "marital-status", "occupation",
                   "relationship", "race", "sex", "capital-gain",
                   "capital-loss", "hours-per-week",
                   "native-country", "income"]

local_file_name = "adult.csv"

print("First try to load the data set from a local file if it exists")
try:
    my_data = pd.read_csv(local_file_name)
except FileNotFoundError:
    print("File not found, loading from the Internet and saving to disk...")
    my_data = pd.read_csv(url, header=None)
    my_data.columns = columns_headers
    # COMMENT THIS LINE IF YOU DON'T WANT TO SAVE FILE LOCALLY
    my_data.to_csv(local_file_name, sep=",", index=False)

print("\nLet's look at the shape of the data frame")
print(my_data.shape)

print("\nLet's look at the first few rows of the date frame")
print(my_data.head())

print("\nLet's look at the data types of the columns in the data frame")
print(my_data.dtypes)

print("\nLet's make histograms of numerical variables and some statistics")

# I'll keep a list of outliers while iterating through columns
outliers_list = []
for category in my_data.dtypes.keys():
    try:
        column = my_data.loc[:, category]
        mean = np.mean(column)
    except TypeError:
        pass
    else:
        plt.hist(column)
        plt.xlabel(category)
        plt.ylabel("frequency")
        plt.show()
        print("\nWhat are some summary stats for the '{}' category?"
              .format(category)
              )
        st_dev = np.std(column)
        limit_hi = mean + 2 * st_dev
        limit_lo = mean - 2 * st_dev
        print("""\tMin_value: {}
        Max_value: {}
        Median: {}
        Mean: {:.3f}
        St. dev.: {:.3f}
        Limit_Low: {:.3f}
        Limit_High: {:.3f}
        """
              .format(np.min(column),
                      np.max(column),
                      np.median(column),
                      mean,
                      st_dev,
                      limit_lo,
                      limit_hi
                      )
              )
        print("How many outliers in this category? (i.e. 2+ st.dev.)")
        flag_bad = (column < limit_lo) | (column > limit_hi)
        if category != "fnlwgt":  # skip this variable 'cos I'll delete it
            outliers_list.append(flag_bad)
        print(sum(flag_bad))

print("\nLet's plot all the numeric columns against each other.")
scatter_matrix(my_data, figsize=[8, 8], s=50)
plt.show()

print("""\n\nREMOVING OUTLIERS.""")

print("\nIf I wanted to remove all outliers, how many rows would I remove?")
print("\nI'll create a new dataframe for outliers with a new calculated column")
print("The last column 'union' is the union of outlier values in each row\n")
new_out_df = pd.DataFrame(outliers_list).T
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

print("\nHow this count would look like if I remove rows with outliers?")
my_data = my_data.loc[~new_out_df["union"], :]
print(pd.crosstab(my_data["income"], columns="count"))

print("\nOkey, the decrease looks roughly proportional in both categories.")
print("Let's keep it that way for now.")
print("Resulting shape, without rows with outliers.")
print(my_data.shape)


print("""\n\nZ-NORMALIZATION""")
print("\nI will z-normalize the 'age' variable using StandardScaler")
age_df = my_data[['age']]
standard_scaler = preprocessing.StandardScaler().fit(age_df)
my_data["age"] = standard_scaler.transform(age_df)
print("z-normalized 'age' column mean and st.dev are now {:.3f} and {:.3f}"
      .format(my_data.age.mean(),
              my_data.age.std()
              )
      )

plt.hist(my_data.loc[:, "age"])
plt.title("Age Category after Z-normalization")
plt.xlabel("normalized age")
plt.ylabel("frequency")
plt.show()


print("""\n\nEQUIVALENT VARIABLES""")
print("""\n\t\tThe dataset contains two equivalent variables,
      'education' and 'education-years'.

      It is easy to see which 'education' value corresponds to which
      'education-years' value by constructing a two-way table of counts
      as follows. For example, 1 is 'Preschool' and 16 is 'Doctorate'.
      """)

print(pd.crosstab(index=my_data["education-years"].astype("str"),
                  columns=my_data["education"]
                  )
      )


print("""\n\nBINNING""")


def eqw_three_bin(variable):
    """Apply equal-width binning with 3 bins to variable (type str)."""
    num_bins = 3
    x = my_data[variable]
    width_bin = (max(x) - min(x))/num_bins
    min_bin_1 = float('-inf')
    max_bin_1 = min(x) + 1 * width_bin
    max_bin_2 = min(x) + 2 * width_bin
    max_bin_3 = float('inf')

    print("\n########\n\n Bin 1 is from ", min_bin_1, " to ", max_bin_1)
    print(" Bin 2 is greater than ", max_bin_1, " up to ", max_bin_2)
    print(" Bin 3 is greater than ", max_bin_2, " up to ", max_bin_3)

    my_data.loc[(min_bin_1 < x) & (x <= max_bin_1), variable] = "L"  # Low
    my_data.loc[(max_bin_1 < x) & (x <= max_bin_2), variable] = "M"  # Med
    my_data.loc[(max_bin_2 < x) & (x < max_bin_3), variable] = "H"  # High


for cat in ["hours-per-week", "capital-gain",
            "capital-loss", "education-years"]:
    print("\nI'll use equal-width binning with 3 bins for", cat)
    eqw_three_bin(cat)
    my_data[cat].value_counts().plot(kind="bar", title=cat)
    plt.show()
    print(my_data[cat].value_counts())


print("""\n\nIMPUTING VALUES.""")
print("""\n\t\tThe column 'native-country' has 583 missing values
      denoted by '?'.
      I will replace '?' with the most frequent value, 'United-States'
      """)

missing_values = my_data.loc[:, "native-country"] == " ?"
my_data.loc[missing_values, "native-country"] = " United-States"

print("""\n\t\tThe column 'workclass' has 1836 missing values
      denoted by '?'.
      I will replace '?' with the most frequent value, 'Private' (22696 counts)
      """)

missing_values = my_data.loc[:, "workclass"] == " ?"
my_data.loc[missing_values, "workclass"] = " Private"


print("""\n\nCONSOLIDATING CATEGORIES""")
print("""\n\t\tI want to consolidate categories in the 'native-country'
      variable into 2 categories, 'US' and 'non-US', because the vast
      majority of cases are 'US' and all other countries account for
      a very small amount of cases.
      """)

non_united_states = my_data.loc[:, "native-country"] != " United-States"
my_data.loc[~non_united_states, "native-country"] = "US"
my_data.loc[non_united_states, "native-country"] = "Non-US"
print(my_data.loc[:, "native-country"].value_counts())

print("""\n\t\tI want to consolidate categories in the 'workclass'
      variable.
      I will group " State-gov", " Federal-gov", " Local-gov" as 'gov'.
      I will group " Private", " Self-emp-not-inc", " Self-emp-inc" , and
      " Without-pay" as 'private'.
      There 14 cases of 'Without-pay' which I include into the most frequent
      category 'Private'.
      I will remove 7 'Never-worked' cases because I don't know what to do
      with them.
      """)

for cat in [" State-gov", " Federal-gov", " Local-gov"]:
    my_data.loc[(my_data.loc[:, "workclass"] == cat), "workclass"] = "gov"

for cat in [" Private", " Self-emp-not-inc", " Self-emp-inc", " Without-pay"]:
    my_data.loc[(my_data.loc[:, "workclass"] == cat), "workclass"] = "private"

print(my_data.loc[:, "workclass"].value_counts())
never_worked_values = my_data.loc[:, "workclass"] == ' Never-worked'
my_data = my_data.loc[~never_worked_values, :]
print(my_data.loc[:, "workclass"].value_counts())

print("""\n\t\tI want to consolidate categories in the 'marital-status'
      variable.
      I will group  ' Never-married', ' Separated', ' Widowed',
      ' Divorced' and ' Married-spouse-absent' into 'unmarried'
      category.
      I will group ' Married-civ-spouse' and ' Married-AF-spouse'
      as 'married' category.
      """)
for cat in [' Never-married', ' Separated',
            ' Widowed', ' Divorced', ' Married-spouse-absent']:
    my_data.loc[(my_data.loc[:, "marital-status"] == cat),
                "marital-status"] = "unmarried"

for cat in [' Married-civ-spouse', ' Married-AF-spouse']:
    my_data.loc[(my_data.loc[:, "marital-status"] == cat),
                "marital-status"] = "married"

print(my_data.loc[:, "marital-status"].value_counts())

print("""\n\t\tI want to consolidate categories in the 'occupation'
      variable.
      I will group  ' Handlers-cleaners', ' Craft-repair', ' Transport-moving',
      ' Farming-fishing' and ' Priv-house-serv' into 'Manual'
      category.

      """)
for cat in [' Handlers-cleaners', ' Craft-repair', ' Transport-moving',
            ' Farming-fishing', ' Priv-house-serv']:
    my_data.loc[(my_data.loc[:, "occupation"] == cat),
                "occupation"] = "Manual"

print(my_data.loc[:, "occupation"].value_counts())


"""REMOVING MISSING VALUES."""
print("""\n\t\tI already replaced missing values in the 'workclass'
      and 'native-country' categories. Now I want to remove rows
      with missing values in the "occupation" category.
      """)

question_marks = my_data.loc[:, "occupation"] == " ?"
print("I will remove", sum(question_marks), "values. The result is:")
my_data = my_data.loc[~question_marks, :]
print(my_data.loc[:, "occupation"].value_counts())


print("""\n\nREMOVING COLUMNS.""")
print("""\n\t\tI want to remove the 'fnlwgt' column since
      it relates to some obscure calculated statistic for
      similarity of people's demographic characteristics within each state.

      I also remove the column 'education' because it has a numerical
      equivalent 'education-years'.
      """)

my_data = my_data.drop(columns="fnlwgt")
my_data = my_data.drop(columns="education")


print("""\n\nONE-HOT ENCODING.""")
print("""\n\t\tI will one-hot encode the 'relationship' variable.
      """)
pd.get_dummies(my_data, columns=["relationship"], prefix="relat")


print("""\n\nMY SUMMARY:
      The dataset had 15 variables -- 6 numeric and 9 non-numerical
      (including the class label - 'income')

      (1)  I removed 5818 rows with outliers from 5 numeric variables
      (out of the total 32561 rows in the data set). I skipped the 'fnlwgt'
      variable here because I'll delete it entirely later. I think such
      removal of outliers did not affect the prediction variable "income"
      because the removal affected both labels (i.e. '>50K' and '<=50K')
      proportionately.

      (2) I z-normalized the 'age' variable using the StandardScaler
      from the sklearn library. This was done as an exercise. I think
      I should probably bin this variable too. I am not yet sure
      how this will be used in subsequent lessons.

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
                          In the 'workclass', I also removed 7 'Never-worked'
                          cases (I prefer to think of them as outliers).
          'marital-status' -> reduction of the number of categories to 2.
          'occupation' -> reduction of the number of categories to 10.
                          In the 'occupation', I also removed rows with
                          missing values.
                          This probably needs to be consolidated further.

      (6) I removed the following columns:
          'fnlwgt': it contained some calculated data which I can't imagine
                     how to use.
          'education': it has a numerical equivalent - 'education-years'.

      (7) I one-hot encoded the 'relationship' variable with 6 categories.
      The meaning of these categories is not fully clear to me. I've found
      that, in general, this column contains data on 'Relationship to Head
      of Household'. So, e.g., 'Not-in-family' means a householder living
      alone or with non-relatives. I'll still have to find out what
      other categories mean exactly.
      """)
