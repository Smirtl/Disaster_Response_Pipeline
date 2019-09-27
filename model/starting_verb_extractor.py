import numpy as np
import pandas as pd

#NLP
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin

#Define StartingVerbExtractor class:
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        try:
            sentence_list = sent_tokenize(text)
        except:
            return 0
        if not sentence_list: #empty lists
            return 0
        for sentence in sentence_list:
            try:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
                return 0
            except:
                return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)