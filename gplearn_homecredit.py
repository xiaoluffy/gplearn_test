from contextlib import contextmanager
import time
import pandas as pd
import numpy as np

from gplearn.genetic import SymbolicTransformer

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 22
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import gc

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def plot_feature_importances(df, threshold=0.9, filename=None):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.

    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances

    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column

    """
    plt.rcParams['font.size'] = 18

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 12))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Feature Importances')

    # Cumulative importance plot

    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features');
    plt.ylabel('Cumulative Importance');
    plt.title('Cumulative Feature Importance');
    if filename:
        plt.savefig(filename)
    plt.show();

    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))


#     return df


def plot_label_corr(df, filename=None):
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(10, 6))

    sns.distplot(df)

    plt.ylabel('distribution');
    plt.xlabel('');
    plt.title('Pearson Correlation Coefficient with label')
    if filename:
        plt.savefig(filename)
    plt.show()


def model(features, test_features, encoding='ohe', n_folds=5):
    """Train and test a light gradient boosting model using
    cross validation.

    Parameters
    --------
        features (pd.DataFrame):
            dataframe of training features to use
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame):
            dataframe of testing features to use
            for making predictions with the model.
        encoding (str, default = 'ohe'):
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation

    Return
    --------
     submission (pd.DataFrame):
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame):
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame):
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.

    """
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    # Extract the labels for training
    labels = features['TARGET']

    # Remove the ids and target
    features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])

    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join='inner', axis=1)

        # No categorical indices to record
        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        # Create a label encoder
        label_encoder = LabelEncoder()

        # List for storing categorical indices
        cat_indices = []

        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)

    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    # Extract feature names
    feature_names = list(features.columns)

    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=False, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective='binary', boosting_type='goss',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=50)

        # Train the model
        model.fit(train_features, train_labels, eval_metric='auc',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], categorical_feature=cat_indices,
                  early_stopping_rounds=100, verbose=200)

        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]

        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return submission, feature_importances, metrics


def test_gp(gp, n_component, train, test, foldername='pg'):

    print('开始 test gp')
    label = train['TARGET']
    x_train = train.drop(columns=['TARGET'])
    x_test = test.copy()

    del train, test
    gc.collect()

    train_idx = x_train.index
    test_idx = x_test.index

    gp.fit(x_train, label)

    gp_train_feature = gp.transform(x_train)
    gp_test_feature = gp.transform(x_test)

    print('gplearn fit transform 结束')

    new_feature_name = [str(i) + 'V' for i in range(1, n_component + 1)]

    del gp_train_feature, gp_test_feature
    gc.collect()

    train_new_feature = pd.DataFrame(gp_train_feature, columns=new_feature_name, index=train_idx)
    test_new_feature = pd.DataFrame(gp_test_feature, columns=new_feature_name, index=test_idx)

    x_train = pd.concat([x_train, train_new_feature], axis=1)
    x_test = pd.concat([x_test, test_new_feature], axis=1)

    del train_new_feature, test_new_feature
    gc.collect()

    print('组合特征结束')

    corr = x_train.corr().abs()
    plt.figure(figsize=(300, 200))
    sns.heatmap(corr, linewidths=.2, cmap="YlGnBu")
    plt.savefig(foldername + 'corr.png')

    print('heatmap 结束')

    label_corr = corr['TARGET']
    label_corr.drop('TARGET', inplace=True)
    plot_label_corr(label_corr, foldername + 'label_corr.png')

    print('label 相关性分析结束')

    x_train = pd.concat([x_train, label], axis=1)

    submission, feature_importances, metrics = model(x_train, x_test)

    print('模型训练结束')

    plot_feature_importances(feature_importances, foldername + 'fi.png')

    print(metrics)


def main():
    with timer('读取文件时间'):
        train = pd.read_csv('train_541.csv', nrows=10000)
        test = pd.read_csv('test_541.csv', nrows=10000)
        print('Training set full shape: ', train.shape)
        print('Testing set full shape: ', test.shape)

    function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'max', 'min']

    gp1 = SymbolicTransformer(generations=1, population_size=1000,
                              hall_of_fame=600, n_components=100,
                              function_set=function_set,
                              parsimony_coefficient=0.0005,
                              max_samples=0.9, verbose=1,
                              random_state=0, n_jobs=3)

    train.fillna('median', inplace=True)
    test.fillna('mdeian', inplace=True)

    print('填充完毕')

    with timer('test pg1'):
        test_gp(gp1, 100, train, test, foldername='pg1')


if __name__ == '__main__':
    gc.enable()
    with timer('time sum'):
        main()