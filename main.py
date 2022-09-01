from engine import load, Recommender

if __name__ == "__main__":
    df, df_content = load(data_dir="./data")

    recommender = Recommender(n_top=10)
    recommender.fit(df, df_content)

    article_recommendations = recommender.predict(user_ids=[1, 10, 100, 10000])
    for user_id, user_recommendation in article_recommendations.items():
        article_ids, article_names = user_recommendation
        print(f"For user_id: {user_id}:")
        print(f"We recommended: {article_names}")
