# SAAS ARTICLES CLUSTERING
Returns clustered articles by merging it prev data if available

## Installation
Install the required libraries

```sh
$ pip install -r requirements.txt
```

## Unit Testing

The various unit tests performed: 

1. Test to ensure function get_twitter_clusters is working correctly.

## Running the tests
To run unit tests - 
```sh
$ pytest path/to/this/repository
```

## Static Code Analysis Tests
Static code analysis examines the code and provides an understanding of the code structure, and can help to ensure that the code adheres to industry standards. 

Below two linters have been for this purpose:

- **flake8**
- **pylint**

To perform code analysis tests using flake8 (will analyse all scripts in the repository):
```sh
$ flake8 --ignore F401, F403, F405, E731, W605, E722, E501 .
```

To perform code analysis tests using pylint:
```sh
$ pylint --output-format=text app/

```


## Deployment of API
Run the server
```sh 
$ python run.py
```
## Hitting the API
Get Clustering
```python

import requests
articles=[]
query_dict_twitter = {"query":query,"source":"twitter","articles": twitter_articles}
query_dict_news = {"query": query, "source": "news", "articles": news_articles}
articles = query_dict_twitter  + query_dict_news
data = {"new_data":articles,"prev_data":{"news":news_rep,"twitter":twitter_rep}}
res1 = requests.post(' http://13.233.204.70:8080/clustering/articles', json = data).json()
```

## Authors
  
* **Ashutosh Vishnoi**
