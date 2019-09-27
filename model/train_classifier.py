import sys

#General
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

#NLP
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

#Machine Learning
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

import pickle

from starting_verb_extractor import StartingVerbExtractor


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Categorized_Messages", con = engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    return X, Y, Y.columns


def tokenize(text):
    tokens = [w for w in word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower())) if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    #Build pipeline using CountVectorizer and Tfidf in feature union with the Starting Verb extractor. Classifier: RandomForest 
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        #('clf', MultiOutputClassifier(RandomForestClassifier()))
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    #Choosing parameters for GridSearch:
    parameters = {
        'features__text_pipeline__vect__ngram_range' : ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__n_estimators': [50],# 100, 200],
        #'features__transformer_weights': (
        #{'text_pipeline': 1, 'starting_verb': 0.5},
        #{'text_pipeline': 0.5, 'starting_verb': 1},
        #{'text_pipeline': 0.8, 'starting_verb': 1},
        #)
    }

    #Perform GridSearch and retrieve best classifier:
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=5)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_frame = pd.DataFrame(Y_pred, columns = category_names)

    for column in category_names:
        print('Classification report for the category "{}":'.format(column))
        print(classification_report(Y_test[column], Y_pred_frame[column]))   
    
    
def save_model(model, model_filepath):
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()