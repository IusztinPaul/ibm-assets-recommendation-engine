from typing import List, Tuple

import pandas as pd


def get_article_names(article_ids: List[str], df: pd.DataFrame) -> List[str]:
    """
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the title column)
    """

    article_names = df[df["article_id"].isin(article_ids)]
    article_names = article_names[["article_id", "title"]]
    article_names = article_names.drop_duplicates()

    unknown_article_names = pd.DataFrame(data=article_ids, columns=["article_id"])
    unknown_article_names["title"] = "unknown"
    unknown_article_names = unknown_article_names[~unknown_article_names["article_id"].isin(article_names["article_id"])]

    article_names = pd.concat([article_names, unknown_article_names], axis=0)
    article_names = article_names.set_index("article_id").loc[article_ids]
    article_names = article_names["title"].tolist()

    return article_names


def get_user_articles(user_id: int, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    INPUT:
    user_id - (int) a user id
    df - (pandas dataframe) user-article interaction DataFrame

    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the doc_full_name column in df_content)

    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    """

    article_ids = df[df["user_id"] == user_id]["article_id"].unique().tolist()
    article_names = get_article_names(article_ids, df)

    return article_ids, article_names
