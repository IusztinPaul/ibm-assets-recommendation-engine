from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from engine import data, utils, rank


def create_content_article_matrix(
        df_content: pd.DataFrame,
        df: pd.DataFrame,
        vectorize_column: str = "doc_full_name"
) -> pd.DataFrame:
    """
    Function that creates a content vector for every article.

    :param df_content: The IBM content article DataFrame.
    :param df: The IBM user-article interaction DataFrame.
    :param vectorize_column: The "df_content" column to use for vectorization
    :return: A DataFrame with all the vectorized articles.
    """

    # df_content DataFrame does not contain all the articles,
    # therefore where it has missing data add the article title from the df interaction DataFrame.
    to_vectorize_df = pd.merge(
        df[["article_id", "title"]].drop_duplicates(subset=["article_id"]),
        df_content[["article_id", vectorize_column]].drop_duplicates(subset=["article_id"]),
        how="outer",
        on="article_id"
    )
    to_vectorize_df[vectorize_column] = to_vectorize_df[vectorize_column].fillna(to_vectorize_df["title"])

    vectorizer = CountVectorizer(tokenizer=data.tokenize)
    vectorized_data = vectorizer.fit_transform(to_vectorize_df[vectorize_column].values)
    vectorized_data = vectorized_data.toarray()
    vectorized_data = pd.DataFrame(data=vectorized_data, columns=vectorizer.get_feature_names_out())

    article_df = pd.concat([
        to_vectorize_df[["article_id"]],
        vectorized_data,
    ], axis=1)
    article_df = article_df.set_index("article_id")

    return article_df


def get_top_unseen_similar_articles(
        article_id: str,
        seen_article_ids: List[str],
        content_article_df: pd.DataFrame,
        df: pd.DataFrame,
        minimum_similarity_score: int = 3
) -> Tuple[list, list]:
    """
    :param article_id: The reference article_id to which the function will find similar content.
    :param seen_article_ids: The article_ids that are already "seen" / not valid.
    :param content_article_df: The IBM articles vectorized content DataFrame.
    :param df:  IBM user-article interaction DataFrame
    :param minimum_similarity_score: The minimum value that we accept as the similarity score
        between a user and an article.

    :return:
        content_recommended_article_ids: a list of similar articles by article id
        content_recommended_article_names: a list of similar articles by article title
    """

    content_recommended_article_ids, content_recommended_article_names = get_top_similar_articles(
            article_id=article_id,
            content_article_df=content_article_df,
            df=df,
            minimum_similarity_score=minimum_similarity_score
        )

    content_recommended_article_ids, content_recommended_article_names = \
        np.array(content_recommended_article_ids), np.array(content_recommended_article_names)
    valid_content_article_ids = np.where(
        ~np.isin(content_recommended_article_ids, seen_article_ids)
    )[0]
    content_recommended_article_ids = content_recommended_article_ids[valid_content_article_ids]
    content_recommended_article_names = content_recommended_article_names[valid_content_article_ids]

    content_recommended_article_ids = content_recommended_article_ids.tolist()
    content_recommended_article_names = content_recommended_article_names.tolist()

    return content_recommended_article_ids, content_recommended_article_names


def get_top_similar_articles(
        article_id: str,
        content_article_df: pd.DataFrame,
        df: pd.DataFrame,
        minimum_similarity_score: int = 3
) -> Tuple[list, list]:
    """
    INPUT:
        article_id: The article_id to which we want to see the most similar content.
        df: DataFrame that contains the raw content information about the articles.
        content_article_df: DataFrame that contains the content article vectors.
        minimum_similarity_score: The minimum similarity score we accept.

    :return:
        articles_ids: a list of similar articles by article id
        articles:  a list of similar articles by article title
    """

    # If the article is new we cannot make any recommendations.
    if article_id not in content_article_df.index:
        return [], []

    current_article_vec = content_article_df.loc[article_id]
    similarities = content_article_df @ current_article_vec
    similarities = similarities.drop(article_id)
    similarities = similarities.sort_values(ascending=False)
    similarities = similarities[similarities >= minimum_similarity_score]

    articles_ids = pd.unique(similarities.index).tolist()
    articles = utils.get_article_names(articles_ids, df)

    return articles_ids, articles


def get_recommendations(
        user_id: str,
        content_article_df: pd.DataFrame,
        df: pd.DataFrame,
        n_count: int = 10,
        minimum_similarity_score: int = 3
) -> Tuple[List[str], List[str]]:
    """
    INPUT:
        user_id: The user to which we want to recommend content to.
        df: DataFrame that contains the raw content information about the articles.
        content_article_df: DataFrame that contains the content article vectors.
        n_count: The number of articles to recommend.
        minimum_similarity_score: The minimum accepted similarity between articles.

    OUTPUT:
        recommendations_ids - (list) a list of recommendations for the user by article id
        recommendations - (list) a list of recommendations for the user by article title
    """

    seen_articles_df = df.groupby(["user_id", "article_id"], as_index=False)["title"].count().rename(
        columns={"title": "views"})
    seen_articles_df = seen_articles_df[seen_articles_df["user_id"] == user_id]
    if len(seen_articles_df) == 0:
        return rank.get_recommendations(n_count, df)

    seen_articles_df = seen_articles_df.sort_values(by=["views"], ascending=False)
    seen_articles_series = seen_articles_df["article_id"]

    recommendations_ids = []
    recommendations = []
    for article_id in seen_articles_series:
        article_ids, article_names = get_top_unseen_similar_articles(
            article_id,
            seen_article_ids=seen_articles_series,
            content_article_df=content_article_df,
            df=df,
            minimum_similarity_score=minimum_similarity_score
        )

        recommendations_ids.extend(article_ids)
        recommendations.extend(article_names)

        if len(recommendations) >= n_count:
            break

    recommendations_ids = recommendations_ids[:n_count]
    recommendations = recommendations[:n_count]

    return recommendations_ids, recommendations
