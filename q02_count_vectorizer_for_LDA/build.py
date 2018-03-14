
# Default imports

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.nlp_day_02_project.q01_load_data_tfidf.build import q01_load_data_tfidf

from sklearn.feature_extraction.text import CountVectorizer


def q02_count_vectorizer_for_LDA(path):
    data, tfidf, tfidf_feature_names = q01_load_data_tfidf(path)
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    matrix = vectorizer.fit_transform(data['talkTitle'])
    feature_names = vectorizer.get_feature_names()
    return matrix, feature_names

