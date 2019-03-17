#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains Milestone 3 Assignment.

Data Model and Evaluation

The data set is Census Income Data Set
Purpose: Predict whether income exceeds $50K/yr based on census data
https://archive.ics.uci.edu/ml/datasets/Census+Income
http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
"""
import os
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import seaborn as sns
sns.set()


def print_hist(a_column, a_category):
    """Print a histogram for a given column (series) and category (str)."""
    plt.hist(a_column)
    plt.title("Histogram for the {} variable".format(a_category))
    plt.xlabel(a_category)
    plt.ylabel("frequency")
    plt.show()


def print_summ_stats(category, a_column, limit_hi, limit_lo, num_outliers):
    """Print summary stats for a given column (a pandas series object)."""
    print("""\nThe '{}' variable:
        Count: {}
        Distinct: {}
        Min_value: {}
        Max_value: {}
        Median: {}
        Mean: {:.3f}
        St. dev.: {:.3f}
        Limit_Low: {:.3f}
        Limit_High: {:.3f}
        # outliers: {:.3f} (2x std from the mean)
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
            st_dev = np.std(column)
            limit_hi = mean + 2 * st_dev
            limit_lo = mean - 2 * st_dev
            flag_bad = (column < limit_lo) | (column > limit_hi)
            if category != "fnlwgt":  # skip 'fnlwgt' var. 'cos I'll delete it
                outliers_list.append(flag_bad)
            if show:
                print_hist(column, category)
                num_outliers = sum(flag_bad)
                print_summ_stats(category, column,
                                 limit_hi, limit_lo,
                                 num_outliers)
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


def print_intro_data_prep():
    """Print a short narrative on the data preparation."""
    print("""\nDATA PREPARATION

    This will include:  - removing outliers
                        - dealing with missing value
                          (a) imputation (most frequent value), and
                          (b) removal
                        - consolidation of categories within variables
                        - removal of a very rare category within a variable
                        - removal of extraneous columns
                        - binning 2 numerical variables
                        - z-normalization of 3 numerical variables
                        - one-hot encoding of the entire data set (except
                          class labels and 3 z-norm'ed numerical variables)
                        - replacing the two class labels with "0" and "1"
                          (i.e. income '<=50K' and '>50K'; for ease of
                          computations only, not required by classifiers)""")


def print_data_prep_conclusion(dataset):
    """Print a description of each attribute after data preparation."""
    print("""\nDATASET DESCRIPTION just before one hot encoding:

            {} attributes, {} observations, no missing values, no outliers

            1) age: numerical, z-normalized
            2) workclass: 2 consolidated categories: 'gov', 'private'
            3) education-years: numerical, z-normalized
            4) marital-status: 2 consolidated categories: married, unmarried
            5) occupation: 10 categories, with 'manual' being a result
                         of consolidating 5 other categories
            6) race: categorial (5 categories), no changes vs. original dataset
            7) sex: categorial (2 categories), no changes vs. original dataset
            8) capital-gain: binned into 3 equal-width bins, L, M, H
            9) capital-loss: binned into 3 equal-width bins, L, M, H
            10) hours-per-week: numerical, z-normalized
            11) native-country: 2 consolidated categories: 'US', 'Non-US'
            12) relationship: 6 categories, 'relationship' means, in
                              general, relationship to the head of household.
            13) income: class label: <=50K, >50K
            """
          .format(dataset.shape[1] - 1, dataset.shape[0]))


def remove_outliers(a_dataset, all_outliers_list):
    """Return the dataset with removed outliers."""
    print("""\n - REMOVING OUTLIERS""")
    dataset_copy = deepcopy(a_dataset)
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
    print("""\n\n - IMPUTING MISSING VALUES
    \n# of missing values (present in non-numerical variables only): {}"""
          .format(count_missing(dataset_copy))
          )
    missing_values = dataset_copy.loc[:, "native-country"] == " ?"
    print("""\n\t\tThe column 'native-country' has {} missing values denoted
          by ' ?'. Replace ' ?' with the most frequent value, 'United-States'
          """.format(sum(missing_values)))
    dataset_copy.loc[missing_values, "native-country"] = " United-States"

    missing_values = dataset_copy.loc[:, "workclass"] == " ?"
    print("""\n\t\tThe column 'workclass' has {} missing values denoted
          by '?'. Replace '?' with the most frequent value, 'Private'"""
          .format(sum(missing_values)))
    dataset_copy.loc[missing_values, "workclass"] = " Private"

    return dataset_copy


def remove_missing(a_dataset):
    """Return the dataset with removed missing values."""
    dataset_copy = deepcopy(a_dataset)
    question_marks = dataset_copy.loc[:, "occupation"] == " ?"
    print("""\n\n - REMOVING MISSING VALUES
    \n\tRemove {} rows with missing values in the "occupation" category"""
          .format(sum(question_marks)))
    dataset_copy = dataset_copy.loc[~question_marks, :]

    return dataset_copy


def consolidate_categories(a_dataset):
    """Return the dataset with consolidated categories."""
    dataset_copy = deepcopy(a_dataset)
    print("""\n\n - CONSOLIDATING CATEGORIES
    \n\tI will consolidate categories in the 'native-country'
    variable into 2 categories, 'US' and 'non-US', because the vast
    majority of cases are 'US' and all other countries account for
    a very small amount of cases.""")

    non_us = dataset_copy.loc[:, "native-country"] != " United-States"
    # Abbreviate ' United-States' as 'US'
    dataset_copy.loc[~non_us, "native-country"] = "US"
    dataset_copy.loc[non_us, "native-country"] = "Non-US"
    print(dataset_copy.loc[:, "native-country"].value_counts())

    print("""\n\t\tConsolidate categories in 'workclass':""")
    replace_list = [" State-gov", " Federal-gov", " Local-gov"]
    print("{} --> 'gov'".format(replace_list))
    dataset_copy.loc[:, "workclass"].replace(replace_list, "gov", inplace=True)

    replace_list = [" Private", " Self-emp-not-inc", " Self-emp-inc"]
    print("{} --> ''private'\n".format(replace_list))
    dataset_copy.loc[:, "workclass"].replace(replace_list,
                                             "private",
                                             inplace=True)
    print(dataset_copy.loc[:, "workclass"].value_counts())

    print("""' Without-pay' will go to the most frequent category 'private'
          and I will remove 'Never-worked' cases""")
    dataset_copy.loc[:, "workclass"].replace(" Without-pay",
                                             "private",
                                             inplace=True)
    never_worked_values = dataset_copy.loc[:, "workclass"] == ' Never-worked'
    dataset_copy = dataset_copy.loc[~never_worked_values, :]
    print(dataset_copy.loc[:, "workclass"].value_counts())

    print("""\n\t\tConsolidate categories in 'marital-status':""")
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

    print("""\n\t\tConsolidate categories in 'occupation':""")
    replace_list = [' Handlers-cleaners', ' Craft-repair', ' Transport-moving',
                    ' Farming-fishing', ' Priv-house-serv']
    print("{} --> 'Manual'\n".format(replace_list))
    dataset_copy.loc[:, "occupation"].replace(replace_list,
                                              "Manual",
                                              inplace=True)
    return dataset_copy


def remove_columns(a_dataset):
    """Return the dataset with removed columns."""
    dataset_copy = deepcopy(a_dataset)

    print("""\n\n - REMOVING COLUMNS
    \n\tI want to remove the 'fnlwgt' column since it relates to some obscure
    calculated statistic for similarity of people's demographic parameters
    within each state.
    \tI also remove the column 'education' because it has a numerical
    equivalent 'education-years'.
    """)

    dataset_copy.drop(columns=["fnlwgt", "education"], inplace=True)

    return dataset_copy


def apply_binning(a_dataset, vars_to_bin):
    """Return the dataset with binned variables."""
    dataset_copy = deepcopy(a_dataset)
    print("""\n\n - BINNING
    \nI'll use equal-width binning with 3 bins for the variables {}"""
          .format(vars_to_bin))

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
    \nI'll z-normalize the variables {}""".format(vars_to_znorm))

    for cat in vars_to_znorm:
        column = dset[[cat]].astype(float)
        dset[cat] = preprocessing.StandardScaler().fit_transform(column)

    return dset


def one_hot_encode(a_dataset, a_prefix_dict):
    """Return the dataset with one-hot-encoded variables."""
    dataset_copy = deepcopy(a_dataset)
    print("""\n\n - ONE-HOT ENCODING""")

    # Extract column headers from the prefix dict
    clmn_headers = list(a_prefix_dict.keys())

    # One-hot encode the variables listed in clmn_headers
    dataset_copy = pd.get_dummies(dataset_copy,
                                  columns=clmn_headers,
                                  prefix=a_prefix_dict)
    print("\nResulting dataset: {} observations, {} attributes"
          .format(dataset_copy.shape[0], dataset_copy.shape[1]))

    return dataset_copy


def print_heatmap(conf_matrix, title, labels):
    """Print a confusion matrix as a heat map, with labels from a list."""
    sns.heatmap(conf_matrix, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Actual labels')
    plt.ylabel('Predicted labels')
    plt.title(title)
    plt.show()


def do_roc_analysis(XX, YY, model, show=True):
    """Return area under curve metric and prob. estimates; plot ROC curve."""
    # y = probability estimates of the positive class
    y = model.predict_proba(XX.values)[:, 1]
    LW = 1.5  # line width for plots
    LL = "lower right"  # legend location
    LC = "darkgreen"  # Line Color
    # False Positive Rate, True Posisive Rate, probability thresholds
    fpr, tpr, th = metrics.roc_curve(YY.astype(int), y)
    AUC = metrics.auc(fpr, tpr)

    if show:
        plt.figure()
        plt.title('ROC curve for Adult Dataset, {} Model'.format(type(model)))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FALSE Positive Rate')
        plt.ylabel('TRUE Positive Rate')
        plt.plot(fpr, tpr, color=LC, lw=LW,
                 label='ROC curve (area = %0.2f)' % AUC)
        # reference line for random classifier
        plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--')
        plt.legend(loc=LL)
        plt.show()

    return AUC, y


def print_init_description(dataset, url):
    """Provide an initial discrtiption for the dataset."""
    print("""
    \nThe data set is Census Income Dataset downloaded from {}
    \nDATA EXPLORATION
    \nThe dataset initially contains {} observations and {} attributes
    \nThe data types of the variables in the original dataset are:
    \n{}"""
          .format(url, dataset.shape[0], dataset.shape[1] - 1, dataset.dtypes))


def get_predicted_test_labels(model_instance, test_XX, prob_threshold):
    """Return predicted labels for the test data."""
    # Use predict_proba method to guess labels for the test set
    YY = (model_instance.predict_proba(test_XX.values)[:, 1] >= prob_threshold)
    YY = YY.astype(int).astype(str)  # Turn booleans into "0s" and "1s"
    return YY


def print_balanced_accuracy_score(actual_YY, predicted_YY, model_name):
    """Print balanced accuracy score."""
    try:
        bas = metrics.balanced_accuracy_score(actual_YY, predicted_YY)
    except AttributeError:
        print("Check if you have the latest sklearn version")
    else:
        print("""\n\tBalanced_accuracy_score for {}: {:.2f}"""
              .format(model_name, bas))


def convert_dataframe(predicted_YY, actual_YY, df_model_name):
    """Return a dataframe from 2 numpy.ndarrays."""
    predicted_YY = predicted_YY.astype(int)
    actual_YY = actual_YY.astype(int)
    labels_dict = {'Predicted': predicted_YY, 'Actual': actual_YY}
    # Create a pandas dataframe from a dict with numpy ndarrays
    predicted_vs_actuals = pd.DataFrame(labels_dict)
    predicted_vs_actuals.name = df_model_name  # Create a name attrubute

    return predicted_vs_actuals


if __name__ == "__main__":
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    source_data_file = "adult.csv"

    columns_headers = ["age", "workclass", "fnlwgt", "education",
                       "education-years", "marital-status", "occupation",
                       "relationship", "race", "sex", "capital-gain",
                       "capital-loss", "hours-per-week",
                       "native-country", "income"]

    adult = load_data(url, source_data_file, columns_headers)

    print_init_description(adult, url)

    print("""
    \nGraphics and statistics for all variables in the original dataset:""")
    # Get outliers for numerical categories and print graphics
    outliers_list = get_outliers(adult, show=True)

    print_intro_data_prep()

    adult = remove_outliers(adult, outliers_list)
    adult = impute_missing(adult)
    adult = remove_missing(adult)
    adult = consolidate_categories(adult)
    adult = remove_columns(adult)

    print("""\n\nUNSUPERVISED LEARNING - k-means
    \n\tTry to make a scatter plot of a numerical attribute 'education-years'
    and a categorical attribute 'marital-status' after k-means analysis
    with 2 clusters (both attributes were z-normalized)

    \tI don't include the predicted cluster labels into supervised learning
    because these labels seem to be useless and meaningless. The same thing
    is true for my attempts with other attributes
    """)
    mari_edu = adult[['marital-status', 'education-years']]
    mari_edu = pd.get_dummies(mari_edu, columns=['marital-status'])
    # Drop one dummy column so that only 2 columns remain
    mari_edu.drop(['marital-status_unmarried'], axis=1, inplace=True)
    # Standardize both columns
    mari_edu_std = stats.zscore(mari_edu[['marital-status_married',
                                          'education-years']])
    # Instantiate a model class object and use its fit method on the array
    kmeans = KMeans(n_clusters=2).fit(mari_edu_std)
    # Get predicted cluster labels
    y_kmeans = kmeans.predict(mari_edu_std)
    plt.scatter(adult.loc[:, 'marital-status'],
                adult.loc[:, 'education-years'],
                c=y_kmeans)
    plt.xlabel('marital-status')
    plt.ylabel('education-years')
    plt.title('Marital status vs education years')
    plt.show()

    # This switch allows to z-normalize several variables or bin them instead
    ZNORMAL = True
    # Probability threshold
    PROB_TH = 0.3
    # Variables to be binned into categories and then one-hot encoded
    binning_vars = ["capital-gain", "capital-loss"]
    # Variables to be z-normalized
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
                   "native-country": "nat_cou"}
    # Variables excluded from one-hot encoding 'cos they are numerical
    add_columns = {"age": "age",
                   "hours-per-week": "hr_wk",
                   "education-years": "ed_yrs"}

    # Apply z-normalization by setting 'ZNORMAL' to True above
    if ZNORMAL:
        adult = apply_znorm(adult, znorm_vars)
    else:
        binning_vars += znorm_vars  # Extend the list of variables to bin
        prefix_dict.update(add_columns)  # Extend the dict to one-hot encode

    adult = apply_binning(adult, binning_vars)

    print_data_prep_conclusion(adult)

    adult_encoded = one_hot_encode(adult, prefix_dict)

    print("Replacing class labels ' <=50K' and ' >50K' as '0' and '1'")
    adult_encoded.loc[:, "income"].replace(' <=50K', "0", inplace=True)
    adult_encoded.loc[:, "income"].replace(' >50K', "1", inplace=True)

    print("""\n\nSUPERVISED LEARNING - Logistic Regression and Decision Tree
    \nPURPOSE: Predict if income exceeds $50K/yr based on census data""")

    print("\n\tSplit the dataset 80/20 into train/test/labelTrain/labelTest")
    y_adult = adult_encoded.loc[:, "income"].values.astype(str)
    x_adult = adult_encoded.drop("income", axis=1)
    X, XX, Y, YY = train_test_split(x_adult,
                                    y_adult,
                                    train_size=0.8,
                                    test_size=0.2,
                                    random_state=0)

    print("""\n\tInstantiate classifier class objects and use their fit
    methods on the train data and predict_proba methods on the test data""")
    C_parameter = 50. / len(X)
    logit_clf = LogisticRegression(C=C_parameter,
                                   multi_class='ovr',
                                   penalty='l1',
                                   solver='saga',
                                   tol=0.1)
    logit_clf.fit(X, Y)

    dtc_clf = DecisionTreeClassifier()
    dtc_clf.fit(X, Y)

    logit_predicted_YY = get_predicted_test_labels(logit_clf, XX, PROB_TH)
    dtc_predicted_YY = get_predicted_test_labels(dtc_clf, XX, PROB_TH)

    print("""
    \tFor both models, create dataframes of predicted and actual values""")
    logit_df = convert_dataframe(logit_predicted_YY, YY, "LOGISTIC REGRESSION")
    dtc_df = convert_dataframe(dtc_predicted_YY, YY, "DECISION TREE")

    print("""\nEVALUATION
    \nPROB_THRESHOLD: {}
    \tI use this threshold because it results in modestly good performance
    in terms of recall scores on both target classes (<=50K and >50K income)
    \tSince the distribution of the target classes in the
    dataset is ~ 76% to 24% (<=50K income vs >50K income, respectively),
    I want to evaluate the models by their ability to predict the rarer
    higher income cases as well as more prevalent lower income cases"""
          .format(PROB_TH))

    print("""\n\tCreate CONFUSION MATRICES from sklearn.metrics module:""")
    for df in (logit_df, dtc_df):
        conf_matrix = metrics.confusion_matrix(df.loc[:, "Actual"],
                                               df.loc[:, "Predicted"])
        true_neg, false_pos, false_neg, true_pos = conf_matrix.ravel()
        accuracy = (true_neg + true_pos) / sum(sum(conf_matrix))
        recall = true_pos / (true_pos + false_neg)
        specificity = true_neg / (true_neg + false_pos)
        print("\n", df.name, "\n", conf_matrix.T)
        print("\taccuracy (total correct predictions): {:.2f}%"
              .format(accuracy * 100))
        print("\tspecificity (correctly labeled low income): {:.2f}%"
              .format(specificity * 100))
        print("\trecall (correctly labeled high income): {:.2f}%"
              .format(recall * 100))

    print("""\nPresent these confusion matrices better in the form of a heat
    map from the seaborn library:""")
    for df in (logit_df, dtc_df):
        conf_matrix = metrics.confusion_matrix(df.loc[:, "Actual"],
                                               df.loc[:, "Predicted"])
        print_heatmap(conf_matrix.T, df.name, ['<=50K', '>50K'])

    # Calculate another score from the metrics module, for imbalanced datasets
    print("""
    For imbalanced datasets, sklean.metrics includes another score,
    balanced_accuracy_score, arithmetic mean of sensitivity (true positive
    rate) and specificity (true negative rate)""")
    for df in (logit_df, dtc_df):
        print_balanced_accuracy_score(df.loc[:, "Actual"],
                                      df.loc[:, "Predicted"],
                                      df.name)

    print("\nA combined report for each model (from metrics module):\n")
    for df in (logit_df, dtc_df):
        print(df.name)
        print(metrics.classification_report(df.loc[:, "Actual"],
                                            df.loc[:, "Predicted"],
                                            target_names=['<=50K', '>50K']))

    print("\nROC ANALYSIS")
    # y - prob.estimates for the positive class
    for model in (logit_clf, dtc_clf):
        AUC, y = do_roc_analysis(XX, YY, model)

        print("\n\tAUC score (using auc function):",
              np.round(AUC, 2))
        print("\tAUC score (using roc_auc_score function):",
              np.round(metrics.roc_auc_score(YY, y), 2))

    print("""\n\nSAVING TRUE AND PREDICTED LABELS TO CSV FILES ON DISK""")
    for df in (logit_df, dtc_df):
        name_file = "{}_labels.csv".format(df.name[:3])
        df.to_csv(name_file, sep=",", index=False)
        if name_file in os.listdir():
            print("\nFile {} saved in {}".format(name_file, os.getcwd()))

    print("""
    \nGENERAL CONCLUSION

    With probability threshold of 3, I achieve recall scores in the
    range of ~65-85% for the positive and negative classes for both models.
    As the probability threshold goes up or down, this strongly affects
    recall scores on the classes, causing them to diverge

    If I don't use z-normalization on three of the five numerical
    variables in my dataset, but rather use binning, I have recall scores
    in the range of ~75-80% for the positive and negative classes for
    both models (for the same prob. threshold 0.3)

    Z-normalization or binning does not affect the AUC score of ~0.87 for
    the Logistic Regression model, but the Decision Tree is better off with
    binning vs z-normalization (~0.85 vs. 0.75, respectively)""")
