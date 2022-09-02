from typing import List, Tuple

import numpy as np
import pandas as pd

from engine import utils


def get_top_sorted_users(user_id: int, df: pd.DataFrame, user_item: pd.DataFrame) -> pd.DataFrame:
    """
    INPUT:
    user_id - (int) The user_id to which we want to find its neighbors.
    df - (pandas dataframe) IBM user-article interaction DataFrame
    user_item - (pandas dataframe) matrix of users by articles:
            1's when a user has interacted with an article, 0 otherwise


    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user
    """

    current_user_vec = user_item.loc[user_id]
    similarity_df = user_item @ current_user_vec
    similarity_df = similarity_df.drop(user_id)
    similarity_df = similarity_df.rename("similarity")
    similarity_df = similarity_df.reset_index()

    user_articles_views_count = df.groupby("user_id")["article_id"].count().rename("num_interactions").reset_index()
    similarity_df = similarity_df.merge(user_articles_views_count, how="left", on="user_id")

    neighbors_df = similarity_df.sort_values(by=["similarity", "num_interactions"], ascending=False)
    neighbors_df = neighbors_df.rename(columns={"user_id": "neighbor_id"})

    return neighbors_df


def get_recommendations(
        user_id: int,
        df: pd.DataFrame,
        user_item: pd.DataFrame,
        n_top: int = 10
) -> Tuple[List[str], List[str]]:
    """
    INPUT:
    user_id - (int) the user id to which we want to make the recommendations
    df - (dataframe) IBM user-article interaction DataFrame
    user_item (dataframe) - matrix of users by articles:
            1's when a user has interacted with an article, 0 otherwise
    n_top - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    """

    similar_users = get_top_sorted_users(user_id, df, user_item)
    seen_article_ids, seen_article_names = utils.get_user_articles(user_id, df)

    recs = {
        "article_id": [],
        "title": [],
    }
    for neighbor_data in similar_users.itertuples():
        article_ids, article_names = utils.get_user_articles(neighbor_data.neighbor_id, df)
        article_ids, article_names = np.array(article_ids), np.array(article_names)

        valid_article_ids = np.where(~np.isin(article_ids, seen_article_ids))[0]
        article_ids = article_ids[valid_article_ids]
        article_names = article_names[valid_article_ids]

        recs["article_id"].extend(article_ids)
        recs["title"].extend(article_names)

        if len(recs["article_id"]) >= n_top:
            break

    if len(recs["article_id"]) > n_top:
        recs = pd.DataFrame(data=recs)
        num_article_views = df.groupby("article_id")["user_id"].count().rename("num_article_views").reset_index()
        recs = recs.merge(num_article_views, how="left", on="article_id")
        recs = recs.sort_values(by="num_article_views", ascending=False)
        recs = recs.iloc[:n_top]

        recs = {
            "article_id": recs["article_id"].values.tolist(),
            "title": recs["title"].values.tolist()
        }

    rec_names = recs["title"]
    recs = recs["article_id"]

    return recs, rec_names
