import sys

from sqlalchemy import create_engine
import pandas as pd

import re
import numpy as np
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'maxent_ne_chunker', 'words'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk, PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import multioutput
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix

import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql('DisasterResponse', engine)
    X = df.message
    y = df.drop(columns=['message', 'id', 'original', 'genre'])
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenization function. Receives raw text as input and then processes the text with these steps:
        1.) Normalizing by converting to all lowercase and removing punctuation
        2.) Splitting text into words or tokens
        3.) Removing words that are too common, also known as stop words
        4.) Identifying different parts of speech and named entities
        5.) Converting words into their dictionary forms, using stemming and lemmatization

    This function then returns the tokenized version of the text
    """
    # Remove punctuation characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ",
                  text.lower())  # Anything that isn't A through Z or 0 through 9 will be replaced by a space

    # Split text into words using NLTK
    words = word_tokenize(text)

    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # Tag parts of speach (PoS) and recognize named entities
    ne_chunk(pos_tag(words))

    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]

    # Reduce words to their root form using default pos
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]
    return lemmed


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # specify parameters for grid search to optimize model
    parameters = {
        'clf__estimator__bootstrap': ['True'],
        'clf__estimator__max_features': ['auto'],
        'clf__estimator__n_estimators': [100]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=5, verbose=2, cv=3)
    return cv


def evaluate_model(cv, X_test, y_test, category_names):
    y_pred = cv.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()