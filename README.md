# IBM Articles Recommendation Engine

# Table of Contents
1. [Motivation](#motivation)
2. [Installation](#installation)
3. [Data](#data)
4. [Usage](#usage)
5. [Licensing, Authors, Acknowledgements](#licensing)

## 1. Motivation <a name="motivation"></a>
The repository started as an educational project around the data IBM provided about user-article interactions 
from their internal blog. Many thanks to their contribution, as they made the data public. Otherwise, this wouldn't have been possible.
The educational part can be seen in the `Recommendations_with_IBM.ipynb` notebook.

After seeing that this is actually useful, we refactored everything under the `engine` Python package. You can quickly
import the `Recommender` class that holds a `scikit` like interface with the `fit` and `predict` methods.

## 2. Installation <a name="installation"></a>
The code was tested under:
* Ubuntu 20.04
* Python 3.9 <br/>
To install all the requirements run (preferably within a virtual environment):
```shell
pip install -r requirements.txt
```
To convert the notebook into `pdf` you need to install the following:
```shell
sudo apt-get install pandoc
```

## 3. Data <a name="data"></a>
We used a user-article interaction public data provided by IBM. It is based on their internal blog.
It consists of two main `csv` files:
* `user-item-interactions.csv`: Record of every user-article interaction.
* `articles_community.csv`: Information about every article from the IBM internal blog.

* You can download the data [here](https://drive.google.com/drive/folders/1XEFmUJoW19MMoL3oDR_CfR6kRc5Kc-ta?usp=sharing).
* For everything to work outside the box, it should be placed at `./data/`.

## 4. Usage <a name="usage"></a>
### File Structure
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

### Instructions
After all the dependencies are installed, and you downloaded the data at `./data` you can run the recommender with:
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

## 5. Licensing, Authors, Acknowledgements <a name="licensing"></a>
The code is licensed under the MIT license. I encourage anybody to use and share the code as long as you give credit to the original author. 
I want to thank IBM for their contribution to making the data available. Without their assistance, I would not have been able to build the engine.

If anybody has machine learning questions, suggestions, or wants to collaborate with me, feel free to contact me 
at p.b.iusztin@gmail.com or to connect with me on [LinkedIn](https://www.linkedin.com/in/paul-iusztin-7a047814a/).
