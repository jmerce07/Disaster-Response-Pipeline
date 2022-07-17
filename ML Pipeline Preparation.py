#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[2]:


# import libraries
from sqlalchemy import create_engine
import pandas as pd

import re
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords','maxent_ne_chunker', 'words'])
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
from sklearn.metrics import accuracy_score,precision_score, f1_score, recall_score, confusion_matrix

import pickle


# In[3]:


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
X = df['message']
Y = df.drop(columns=['message','id','original','genre'])
Y.head()


# In[4]:


df.head()


# ### 2. Write a tokenization function to process your text data

# In[5]:


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
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Anything that isn't A through Z or 0 through 9 will be replaced by a space
    
    #Split text into words using NLTK
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


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[6]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[7]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y)

# train classifier
pipeline.fit(X_train, y_train)

# predict on test data
y_pred = pipeline.predict(X_test)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[8]:


for i in range(36):
    print(y_test.columns[i], ':')
    print(classification_report(y_test.iloc[:,i], y_pred[:,i], target_names=Y.columns), '________________________________________________________________');


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[9]:


pipeline.get_params()


# In[12]:


# specify parameters for grid search to optimize model
parameters = {
    'clf__estimator__bootstrap': ['True'],
    'clf__estimator__max_features': ['auto'],
    'clf__estimator__n_estimators': [100]    
}

# create grid search object
cv = GridSearchCV(pipeline, parameters)


# In[13]:


cv.fit(X_train, y_train)


# In[14]:


# the best paramesters based on grid search
print(cv.best_params_)

#{'clf__estimator__bootstrap': 'True', 'clf__estimator__max_features': 'auto', 'clf__estimator__n_estimators': 100}


# In[15]:


# build new model with best parameters
model = cv.best_estimator_
print (cv.best_estimator_)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[16]:


y_pred = model.predict(X_test)

for i in range(36):
    print(y_test.columns[i], ':')
    print(classification_report(y_test.iloc[:,i], y_pred[:,i], target_names=Y.columns), '________________________________________________________________');


# In[17]:


accuracy = (y_pred == y_test).mean()
accuracy


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[14]:


from sklearn.naive_bayes import MultinomialNB

nb = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
               ('tfidf', TfidfTransformer()),
               ('clf', MultiOutputClassifier(MultinomialNB())),
])


# In[15]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y)

# train classifier
nb.fit(X_train, y_train)

# predict on test data
y_pred = nb.predict(X_test)


# In[16]:


for i in range(36):
    print(y_test.columns[i], ':')
    print(classification_report(y_test.iloc[:,i], y_pred[:,i], target_names=Y.columns), '________________________________________________________________');


# ### 9. Export your model as a pickle file

# In[17]:


# save the model to disk
filename = 'DisasterResponse_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[18]:


# save the model to disk
filename = 'DisasterResponse_NB_model.sav'
pickle.dump(nb, open(filename, 'wb'))


# In[18]:


with open('classifier.pkl', 'wb') as file:
    pickle.dump(model, file)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




