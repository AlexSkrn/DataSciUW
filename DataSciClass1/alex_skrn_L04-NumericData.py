#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:58:38 2018

@author: alexskrn

This files contains Lession 04 Assignment.
Numeric Data.
Exploratory Data Analysis

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

# I remove the "fnlwgt" category
my_data = my_data.drop(columns="fnlwgt")

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
print("Resulting shape, with 1 less column and without rows with outliers.")
print(my_data.shape)

print("\nLet's try to z-normalize each numerical column (z = (x - mu) / std)")
for cat in my_data.dtypes.keys():
    try:
        my_data[cat].mean()  # check if column is numerical or not
    except TypeError:
        pass
    else:
        my_data[cat] = ((my_data[cat] - my_data[cat].mean())
                        / my_data[cat].std()
                        )
print("Did this work?")
norm_age_mean = np.mean(my_data.loc[:, "age"])
norm_age_std = np.std(my_data.loc[:, "age"])
print("z-normalized 'age' column mean and st.dev are now {:.3f} and {:.3f}"
      .format(norm_age_mean, norm_age_std))

plt.hist(my_data.loc[:, "age"])
plt.xlabel("normalized age")
plt.ylabel("frequency")
plt.show()

print("""\nMy summary:
      Only categorical variables had missing values. I kept them as is for now.
      I removed the 'fnlwgt' category the meaning of which is obscure.
      All numerical categories had outliers.
      I removed rows that had outliers across all such categories.
      I guess this did not affect the prediction variable "income"
      because the removal affected both labels (i.e. '>50K' and '<=50K')
      proportionately
      I also did what I believe is called z-normalization on numerical data.
      """)
