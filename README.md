# Unimelb-COMP90019-Project
[TOC]
## Tweets Sentiment Analysis in Market Movement Prediction 
### Based language, framework and softwares used
#### Language
- Python
- Javascript
- Html5
- Css3

#### Framework & packages
- Scrapy
- Echart.js
- Bootstrap
- Pandas
- Numpy
- Scikit-learn
- PyQuery
- ...

#### Software
- Sublime Text3
- iTerm
- MongoDB
- Anaconda
- Jupyter Notebook


### Data Spider
using Python, Scrapy spider framwork, MongoDB for crawling tweets.

To start the spider, you can refer to the following shell commands:

```bash
cd ./data_spider

scrapy crawl t1TweetSipder 
-a company="Amazon"  # related company
-a query="$AMZN since:2019-01-01 until:2019-01-2"  # keys words and time range
-a lang="en" -s LOG_FILE=./log/spider.log # log to files
```

### Data Analysis

#### Testing models list
- svm
- navie bayes
- random forest
- logistic regression
- k nearest neighbors
- decision tree

#### Scripts

Mainly using the following scripts in data_analysis:

```
sentiment_analysis.py # clean the raw tweets & do sentiment analysis 
predict_and_test_all.py # testing & training datas
```

### Data visualisation
Use Bootstrap and Echart.js to show datas with html files.





