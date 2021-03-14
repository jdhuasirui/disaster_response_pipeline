# Import Python libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

# download NLTK packages
nltk.download('punkt')
nltk.download('stopwords')

# Import NLTK libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Import sklearn libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    '''
    load data from database

    Args:
    database_filepath str: SQLite database file path.

    Returns:
    X dataframe: features dataframe
    Y dataframe: target dataframe
    category_names list: labels of categories
    '''
    # read data from SQLite database
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    
    # Get features and target dataframe
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    category_names = list(Y.columns.values)
    
    return X, Y, category_names


def tokenize(text):
    '''
    process the text data using tokenization

    Args:
    text str: Messages as text data

    Returns:
    list of words after tokenization
    '''
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)

    #lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()

    stopWords = set(stopwords.words('english'))
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stopWords:
            clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    Build a machine learning pipeline with GridSearchCV

    Args:
    None

    Returns:
    Trained model after performing grid search
    '''
    # model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # hyper-parameter grid
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 60, 70]
    }
    # create model
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model using test data

    Args:
    model: trained model
    X_test: Test features
    Y_test: Test target
    category_names: Target labels

    Returns:
    print model evalution results
    '''
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred,columns=category_names)
    for col in category_names:
        print(f'Column Name:{col}\n')
        print(classification_report(Y_test[col],Y_pred[col]))


def save_model(model, model_filepath):
    '''
    Export the model to a pickle file
    '''
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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