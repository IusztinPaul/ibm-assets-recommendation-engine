from typing import Tuple, List

import pandas as pd


def get_recommendations(n_top: int, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    '''
    INPUT:
    n_top - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''

    top_article_ids = get_top_article_ids(n_top, df)
    top_articles = df[df["article_id"].isin(top_article_ids)]
    top_articles = top_articles[["article_id", "title"]]
    top_articles = top_articles.drop_duplicates()
    top_articles = top_articles["title"].tolist()

    return top_article_ids, top_articles


def get_top_article_ids(n_top: int, df: pd.DataFrame) -> list:
    '''
    INPUT:
    n_top - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''

    top_articles = df.groupby("article_id")["user_id"].count().sort_values(ascending=False).iloc[:n_top]
    top_articles = top_articles.index.tolist()

    return top_articles
