
# Disaster Response - ETL/Machine Learning Pipelines and Web Application

## Project Motivation

This App displays data engineering skills used in data science to analyze disaster data from Appen (formally Figure 8) to build a model that classifies disaster messages.

In the "data" folder, you'll find data containing real messages that were sent during disaster events. With this data, a machine learning pipeline was created to categorize messages in order to send the messages to the appropriate disaster relief agency.

The final result is a web app that allows a user to input a new message and get classification results in several categories. Also, the web app will display visualizations of the data. 

## Installations

#### This project requires Python 3. Also the following Python libraries need to be installed:

> - NumPy
> - Pandas
> - Matplotlib
> - Json
> - Plotly
> - Nltk
> - Flask
> - Sklearn
> - Sqlalchemy
> - Sys
> - Re
> - Pickle

## File Descriptions

- **ETL Pipeline Preparation.py**: The first part of your data pipeline is the Extract, Transform, and Load process. This program reads the data, cleans the data, and then stores the data in a SQLite database. This process is ultimately in the final ETL script, process_data.py.

- **process_data.py**: This code extracts data from the CSV file containing the message data and the CSV file containing the classes of the messages. An SQLite database containing a merged and cleaned version of this data is created in the end.

- **ML Pipeline Preparation.py**: For the machine learning pipeline, the data is tokenized, split into a training set and a test set, then uses NLTK and GridSearchCV to create the pipeline that outputs a final (optimized) model. The model predicts classifications for 36 categories (multi-output classification). Finally, the model gets exported to a pickle file which is essentially in train_classifier.py. that automates the model fitting process.

- **train_classifier.py**: This code takes the SQLite database produced by process_data.py, uses it as input, and then uses the data to train a ML model for categorizing messages. The output is a pickle file which contains the optimized fitted model. 

- **disaster_messages.csv**: contains real messages that were sent during disaster events.

- **disaster_categories.csv**: contains the categories for the messages that were sent during disaster events and indicates which category(ies) the messages belong to.

- **templates** folder: This folder contains all of the files necessary to run and render the web app.

## Instructions

In order to create the data base, create the model, and get the web app to work, complete the following steps in order:

1. Run python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db (clean the data and create database)
2. Run python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl (train the data and save the model)
3. Run app.py while in the app's directory (runs the web app)

## Licensing, Authors, Acknowledgements

Thank you to Appen (formally Figure 8) for making this data available and thank you to Udacity for such a great project with real world application. The contents of this repository can be used and expounded upon. Please cite me, Udacity, and Appen as you see fit.
