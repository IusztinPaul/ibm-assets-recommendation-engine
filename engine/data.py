import re
from pathlib import Path
from typing import Tuple, Any, Dict

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])


def load(data_dir: str = "./data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the IBM data.

    :param data_dir: Root data directory.
    :return: A tuple with the user-item DataFrame and with the article content DataFrame.
    """

    data_dir = Path(data_dir)

    df = pd.read_csv(data_dir / "user-item-interactions.csv")
    df_content = pd.read_csv(data_dir / "articles_community.csv")

    del df['Unnamed: 0']
    del df_content['Unnamed: 0']

    df["article_id"] = df["article_id"].transform(normalize_article_id_name)
    df_content["article_id"] = df_content["article_id"].transform(normalize_article_id_name)

    user_ids = map_email_to_id(df)
    df["user_id"] = user_ids
    del df["email"]

    return df, df_content


def normalize_article_id_name(article_id: Any) -> str:
    """
    :param article_id: The article id in any format.
    :return: the normalized article_id as a string representation of a float number.
    """

    return str(float(article_id))


def map_email_to_id(df: pd.DataFrame):
    """
    :param df: The user-article interaction DataFrame.
    :return: A list of user_ids extracted from the "email" column.
    """

    coded_dict = dict()
    counter = 1
    email_encoded = []

    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = counter
            counter += 1

        email_encoded.append(coded_dict[val])

    return email_encoded


def tokenize(text):
    """Function to tokenize an article title

    INPUT:
        text (str) The text to tokenize.

    OUTPUT:
        tokens (list) a list of words
    """

    # Normalize.
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize.
    tokens = word_tokenize(text)

    # Clean and lemmatize.
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stopwords.words("english")
    ]

    # To avoid the dimensionality curse, filter all the words with a length less than 2.
    tokens = [token for token in tokens if len(token) > 2]

    return tokens


def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns

    OUTPUT:
    user_item - user item matrix

    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
    an article and a 0 otherwise
    '''

    user_item = df.groupby(["user_id", "article_id"]).count().unstack()
    user_item.columns = user_item.columns.droplevel()
    user_item[user_item.notna()] = 1
    user_item = user_item.fillna(0)
    user_item = user_item.astype("int64")

    return user_item


def create_test_train_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df.head(40000)
    df_test = df.tail(5993)

    return df_train, df_test


def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe

    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe
                    (unique users for each row and unique articles for each column)
    '''

    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)

    return user_item_train, user_item_test


def compute_split_diff(
        user_item_train: pd.DataFrame,
        user_item_test: pd.DataFrame
) -> Dict[str, np.ndarray]:
    old_users = np.intersect1d(user_item_test.index.values, user_item_train.index.values)
    new_users = np.setdiff1d(user_item_test.index.values, user_item_train.index.values)

    old_articles = np.intersect1d(user_item_test.columns.values, user_item_train.columns.values)
    new_articles = np.setdiff1d(user_item_test.columns.values, user_item_train.columns.values)

    return {
        "old_users": old_users,
        "new_users": new_users,
        "old_articles": old_articles,
        "new_articles": new_articles
    }


def compute_matrix_density(user_item_df: pd.DataFrame) -> float:
    """
    Given a user-item interaction matrix compute the density of the matrix as a percentage.

    INPUT
    user_item_df - user-item interaction DataFrame

    OUTPUT
    density - density as a percentage
    """

    num_ones = user_item_df.sum().sum()
    num_total = user_item_df.shape[0] * user_item_df.shape[1]
    density = num_ones / num_total

    return density * 100
