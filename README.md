# ibm-assets-recommendation-engine

# Motivation

# Install
The code was tested under:
* Ubuntu 20.04
* Python 3.9
To install all the requirements run (preferably within a virtual environment):
```shell
pip install -r requirements.txt
```

# Data
We used a user-article interaction public data provided by IBM. It is based on their blog.
It consists of two main `csv` files:
* `user-item-interactions.csv`: Record of every user-article interaction.
* `articles_community.csv`: Information about every article from the IBM internal blog.

* You can download the data [here](https://drive.google.com/drive/folders/1XEFmUJoW19MMoL3oDR_CfR6kRc5Kc-ta?usp=sharing).
* For everything to work outside the box it should be placed at `./data/`.

# File Structure
The engine was developed within the `Recommendations_with_IBM.ipynb` notebook. It contains the following components:
* EDA
* rank recommendation
* collaborative filtering recommendation
* content-based recommendation
* matrix factorization recommendation (with SVD)

All the logic was refactored and moved into the `engine` Python package which reflects the same logic/structure, except:
* `data.py`: Where all the logic about data loading and different transformations is kept.
* `utils.py`: Where all the utility functions can be found
* `recommender.py`: Where the main class `Recommender` can be found.

# Usage
After installed all the dependencies, and you downloaded the data at `./data` you can run the recommender with:
```shell
python main.py
```

You can easily ship the `Recommender` class within your own code. It follows the `scikit` standards:
```python
from engine import load, Recommender

df, df_content = load()

recommender = Recommender(n_top=10)
recommender.fit(df, df_content)
article_recommendations = recommender.predict(user_ids=[1, 10, 100, 10000])
```

# License..