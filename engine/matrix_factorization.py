from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from engine import data, utils


def compute_svd(user_item_train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_train, s_train, vt_train = np.linalg.svd(user_item_train)

    return u_train, s_train, vt_train


def fit_svd(
        df: pd.DataFrame,
        plot: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_train, df_test = data.create_test_train_split(df)
    user_item_train, user_item_test = data.create_test_and_train_user_item(df_train, df_test)

    split_diff = data.compute_split_diff(user_item_train, user_item_test)

    u_train, s_train, vt_train = compute_svd(user_item_train)

    # Find train indices for all the test users that we can predict.
    old_test_users_train_indices = np.where(
        np.isin(user_item_train.index, split_diff["old_users"])
    )[0]
    # Find train indices for all the test articles we can predict.
    old_test_articles_train_indices = np.where(
        np.isin(user_item_train.columns, split_diff["old_articles"])
    )[0]

    # Keep only test data that is overlapped with the training data (aka old_users and old_articles).
    old_user_item_test = user_item_test.loc[split_diff["old_users"], split_diff["old_articles"]]
    # Filter the u rows and vt columns that reflect the users and articles that are in the test split.
    u_train_old_users = u_train[old_test_users_train_indices, :]
    vt_train_old_articles = vt_train[:, old_test_articles_train_indices]

    sum_errs_train = []
    sum_errs_test = []
    num_latent_features_lookup_range = np.arange(10, s_train.shape[0] + 1, 20)
    for k in num_latent_features_lookup_range:
        # restructure with k latent features
        s_new_train, u_new_train, vt_new_train = np.diag(s_train[:k]), u_train[:, :k], vt_train[:k, :]
        s_new_test, u_new_test, vt_new_test = np.diag(s_train[:k]), u_train_old_users[:, :k], vt_train_old_articles[:k,
                                                                                              :]

        # take dot product
        user_item_train_est = np.around(np.dot(np.dot(u_new_train, s_new_train), vt_new_train))
        user_item_test_est = np.around(np.dot(np.dot(u_new_test, s_new_test), vt_new_test))

        # compute error for each prediction to actual value
        diffs_train = np.subtract(user_item_train.values, user_item_train_est)
        diffs_test = np.subtract(old_user_item_test.values, user_item_test_est)

        # total errors and keep track of them
        err_train = np.sum(np.sum(np.abs(diffs_train)))
        sum_errs_train.append(err_train)

        err_test = np.sum(np.sum(np.abs(diffs_test)))
        sum_errs_test.append(err_test)

    sum_errs_train = np.array(sum_errs_train)
    sum_errs_test = np.array(sum_errs_test)

    if plot:
        plt.plot(num_latent_features_lookup_range, 1 - sum_errs_train / (user_item_train.shape[0] * user_item_train.shape[1]),
                 label="train")
        plt.plot(num_latent_features_lookup_range, 1 - sum_errs_test / (old_user_item_test.shape[0] * old_user_item_test.shape[1]),
                 label="test")
        plt.xlabel('Number of Latent Features')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Number of Latent Features')
        plt.legend()
        plt.show()

    # Now choose the best number of latent features.
    best_num_latent_features = num_latent_features_lookup_range[np.argmin(sum_errs_test)]
    s_fitted_train = np.diag(s_train[:best_num_latent_features])
    u_fitted_train = u_train[:, :best_num_latent_features]
    vt_fitted_train = vt_train[:best_num_latent_features, :]

    return u_fitted_train, s_fitted_train, vt_fitted_train


def get_recommendations(
        user_id: int,
        df: pd.DataFrame,
        user_item: pd.DataFrame,
        fitted_svd_matrices: Tuple[np.ndarray, np.ndarray, np.ndarray],
        n_top: int = 10
) -> Tuple[List[str], List[str]]:
    '''
    INPUT:
    user_id - (int) a user id
    df -
    user_item_train -
    fitted_svd_matrices -
    n_top - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    '''

    user_id_svd_index = np.where(np.isin(user_id, user_item.index))[0]
    u, s, v = fitted_svd_matrices

    u_user = u[user_id_svd_index, :]
    user_article_interactions = u_user @ s @ v
    user_article_interactions = pd.Series(data=user_article_interactions[0, :], index=user_item.columns)
    user_article_interactions = user_article_interactions.sort_values(ascending=False)

    seen_articles_ids, seen_article_names = utils.get_user_articles(user_id, df)
    user_article_interactions = user_article_interactions[~user_article_interactions.index.isin(seen_articles_ids)]
    # user_article_ids_interactions = np.setdiff1d(user_article_interactions.index, seen_articles_ids)
    user_article_ids_interactions = user_article_interactions[:n_top].index.tolist()

    user_article_names_interactions = utils.get_article_names(user_article_ids_interactions, df)

    return user_article_ids_interactions, user_article_names_interactions
