# Disaster Response Pipeline

## Description and Motivation
The goal of this project is to provided an ETL and ML pipeline that takes in disaster messages with labels among 36 message categories that trains a model to predict the categories for unknown messages for the purpose of swift disaster response.

## Data
There are two files used for this project:
* `disaster_messages.csv` It contains raw messages and their English translation, ordered by an *id* column as the unique identifier.  
* `disaster_categories.csv` It contains info about 36 categories. For each unique identifier in the *id* column (which matches the one from the `messages.csv` file), it associates the case to one or more categories in the *categories* column.

The data is not provided with this GitHub Repository.

## Files

### Main Python Scripts
* `process_data.py` - Script containing the ETL pipeline for data preprocessing  
* `train_classifier.py` -  Script containing the ML pipeline building a classification model  
* `run.py` - Script running a Flask module that opens a port to the visualization webpage.
* `README.md` - This readme file  

### Additional Python Scripts
* `starting_verb_extractor.py` - NLP script to extract starting verbs. It is used by the `train_classifier.py` script and the `run.py` script.

### HTML scripts used by `run.py`
* `go.html`
* `master.html`

## Functionality
The `process_data.py` script is run with three arguments: paths to the messages and categories files as well as a path to the database the resulting table should be saved in. Example:  
`python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`  
In the given database file path, a resulting table named 'Categorized_Messages' will be saved.

The `train_classifier.py` script is run next and will build and train a model using grid search. Finally, a pickle file called `classifier.pkl` will be saved.

The `run.py` script will use the database and the pickle file to create the flask web application.

To view the webpage, run all three scripts. Then, in the command line, type
`env|grep WORK` to create an environment.
Use the SPACEID and SPACEDOMAIN, replace the corresponding parts in the URL below to go to the webpage.  

URL: `https://SPACEID-3001.SPACEDOMAIN`

## Python Libraries
Libraries for the `process_data.py` script:

* `import numpy as np`  
* `import pandas as pd`  
* `from sqlalchemy import create_engine`  
* `import pickle`  

Libraries for the `train_classifier.py` script:

General libraries for data handling and loading:
* `import numpy as np`  
* `import pandas as pd`  
* `from sqlalchemy import create_engine`   

NLP libraries:  
* `import nltk`  
* `nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])`  
* `from nltk.tokenize import word_tokenize, sent_tokenize`  
* `from nltk.stem import WordNetLemmatizer`  
* `from nltk.corpus import stopwords`  
* `import re`  

ML libraries:
* `from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier`
* `from sklearn.multioutput import MultiOutputClassifier`
* `from sklearn.model_selection import train_test_split`
* `from sklearn.pipeline import Pipeline, FeatureUnion`
* `from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer`
* `from sklearn.model_selection import GridSearchCV`
* `from sklearn.metrics import classification_report`
* `from sklearn.base import BaseEstimator, TransformerMixin`

* `import pickle`

Starting Verb Extractor:
* `from starting_verb_extractor import StartingVerbExtractor`

Libraries for the `train_classifier.py` script:

* `import json`
* `import plotly`
* `import pandas as pd`

* `from nltk.stem import WordNetLemmatizer`
* `from nltk.tokenize import word_tokenize`

* `from flask import Flask`
* `from flask import render_template, request, jsonify`
* `from plotly.graph_objs import Bar`
* `from sklearn.externals import joblib`
* `from sqlalchemy import create_engine`

* `from starting_verb_extractor import StartingVerbExtractor`

## Output
The website has two main functionalities. The main pages shows an overview over the data with plotly graphs. The search bar allows for ad hoc analysis of a disaster message showing the related categories.#
Have a look at the `screenshot.jpg` to get an idea of the visuals.

## Data Considerations
Since there aren't a lot of data points for several of the disaster categories,
the precision on these categories is very low, resulting in several categories not being identified correctly.
