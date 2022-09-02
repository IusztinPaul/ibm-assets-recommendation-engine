from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from engine import matrix_factorization, data, rank, content


class Recommender:
    """
    IBM user-article recommendation engine.
    """

    def __init__(
            self,
            n_top: int
    ):
        """

        :param n_top: The maximum number of articles recommended to a user.
        """

        self.n_top = n_top

        # Learnable information.
        self.df: Optional[pd.DataFrame] = None
        self.df_content: Optional[pd.DataFrame] = None
        self.df_content_vectors: Optional[pd.DataFrame] = None
        self.user_item: Optional[pd.DataFrame] = None
        self.user_ids: Optional[np.ndarray] = None

        self.u: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        self.vt: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, df_content: pd.DataFrame) -> "Recommender":
        """
        Fit the recommender on your own IBM article data.

        :param df: IBM user-article interaction DataFrame.
        :param df_content: IBM article content DataFrame.
        :return: a reference to the "Recommender" fitted instance.
        """

        self.df = df
        self.df_content = df_content
        self.df_content_vectors = content.create_content_article_matrix(self.df_content, self.df)
        self.user_item = data.create_user_item_matrix(self.df)
        self.user_ids = self.df["user_id"].unique()

        self.u, self.s, self.vt = matrix_factorization.fit_svd(df, plot=False)

        return self

    def predict(self, user_ids: List[int]) -> Dict[int, Tuple[List[str], List[str]]]:
        """
        Recommend articles for users.

        :param user_ids: A series of user_ids to which to recommend articles.
        :return: A dictionary with keys as user_ids and values as a tuple of
            "self.n_top" recommended (article_ids, article_names).
        """

        recommendations = {}
        for user_id in user_ids:
            if user_id not in self.user_ids:
                # If the user is new, use rank recommendations.
                recommendations[user_id] = rank.get_recommendations(
                    n_top=self.n_top,
                    df=self.df
                )
            else:
                recommendations[user_id] = matrix_factorization.get_recommendations(
                    user_id=user_id,
                    df=self.df,
                    user_item=self.user_item,
                    fitted_svd_matrices=(self.u, self.s, self.vt),
                    n_top=self.n_top
                )
                recommended_article_ids, recommended_article_names = recommendations[user_id]
                recommended_article_ids = recommended_article_ids[:8]
                if len(recommended_article_ids) < self.n_top:
                    # If with SVD we haven't predicted enough articles, use content recommendation to fill the gaps.
                    for article_id in recommended_article_ids:
                        content_recommended_article_ids, content_recommended_article_names = \
                            content.get_top_unseen_similar_articles(
                                article_id=article_id,
                                seen_article_ids=recommended_article_ids,
                                content_article_df=self.df_content_vectors,
                                df=self.df,
                                minimum_similarity_score=1  # Put a lower similarity score to add novelty.
                            )

                        recommended_article_ids.extend(content_recommended_article_ids)
                        recommended_article_names.extend(content_recommended_article_names)

                        if len(recommended_article_ids) > self.n_top:
                            break

                    recommended_article_ids = recommended_article_ids[:self.n_top]
                    recommended_article_names = recommended_article_names[:self.n_top]
                    recommendations[user_id] = recommended_article_ids, recommended_article_names

        return recommendations
