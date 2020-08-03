import sys
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import pandas as pd
import numpy as np
import time
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report #multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    # load data from database
    '''
    INPUT:
    database_filepath - (string) a valid filepath to a messages database file to load
    
    OUTPUT:
    X - (df) a dataframe of the messages to classify
    Y - (df) a dataframe of the message categories
    category_names - (list)
    '''
    create_engine_argument = 'sqlite:///{}'.format(database_filepath)
    print(create_engine_argument)
    engine = create_engine(create_engine_argument)
    df = pd.read_sql_table('messages', engine)
    print(df.shape)
    
    X = df.message
    Y = df.drop(columns=['message','genre', 'id', 'original'], axis=1) 
    category_names = Y.columns
    return X, Y, category_names
    


def tokenize(text):
    '''
    INPUT:
    text - text to clean and lemmatize
        
    OUTPUT:
    clean_tokens - cleaned and lemmatized text
    '''
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    OUTPUT:
    model - GridSearchCV object to classify messages
    '''
   
    pipeline = Pipeline([
    ('text_pipeline', Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
    ])),

    ('mo_clf', MultiOutputClassifier(RandomForestClassifier()))
    #('mo_clf', SVC())
    ])
    
    #Parameters to use in GridSearch optimisation
    parameters = {
    'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    'text_pipeline__vect__max_features': (None, 5000, 10000),
    'text_pipeline__tfidf__use_idf': (True, False),
    'mo_clf__estimator__n_estimators': [5, 10],
    'mo_clf__estimator__min_samples_split': [2, 4]
    }

    start = time.process_time()
    model = GridSearchCV(pipeline, param_grid=parameters,  n_jobs=-1, verbose=10, scoring='f1_macro', cv=None)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - SciKit Learn model
    X_test - test messages to classify
    Y_test - categories of messages being classified
    category_names - list of message categories
    
    Outputs classification report
    '''
    Y_pred = model.predict(X_test)
    
    #iterate over columns
    for col_num in range(Y_pred.shape[1]):
        print('Classification Report For Class: ' + category_names)
        report = classification_report(Y_test.values[:,col_num], Y_pred[:,col_num])
        print(report)

    pass


def save_model(model, model_filepath):
    '''
    INPUT:
    model - the model to save
    model_filepath - directory to save model to
        
    Saves model as pickle file in directory referenced by model_filepath
    '''
    pickle.dump(model,open(model_filepath,'wb'))
    
    #pass


def main():
    '''
    Run the classifier
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(dir())
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