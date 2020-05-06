import re
import sys
import pickle
import pandas as pd

from nltk import word_tokenize, download as nltk_download
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sqlalchemy import create_engine

nltk_download('punkt')
nltk_download('stopwords')
nltk_download('wordnet')


def load_data(database_filepath):
    """
    Load the cleaned data from a sql database file.

    Args:
        database_filepath: path for sql dataset.
    Returns:
        X: messages raw text.
        Y: categories cleaned.
        category_names: list of category names.
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    df = pd.read_sql_table('messages', engine, index_col="id")

    category_names = df.columns[4:]
    X = df.message.values
    Y = df[category_names].values
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize a text string by converting to lower, removing ponctuation, 
    spliting in wordss and applying a Lemmatizer and Stemmer.
    
    Args:
        text: raw text string.
    Returns:
        words: list of lemmatize and stemed words.
    """
    # normalize the text removing ponctuation and making it lowercase
    normalized = re.sub("[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize the normalized text
    words = word_tokenize(normalized)
    
    # remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # lemmetatization
    wordnet = WordNetLemmatizer()
    words = [wordnet.lemmatize(w) for w in words]
    
    # stemming
    porter = PorterStemmer()
    words = [porter.stem(w) for w in words]
    
    return words


def build_model():
    """
    Generates a sklearn gridsearch with a pipeline.
    The pipeline has 3 steps: a CountVectorizer, a TfidfTransformer and a MultiOutputClassifier 
    with a RandomForestClassifier.

    Returns:
        pipeline: sklearn pipeline object.
    """
    # build a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,2), max_df=0.75, max_features=10000)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())) 
    ])

    # params dict to tune a model
    # a full version of this grid search is available in the notebook ML Pipeline
    parameters = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__min_samples_split': [2, 3]
    }

    # instantiate a gridsearchcv object with the params defined
    cv =  GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model and prints the f1-score, precision and recall for the X_test and Y_test passed.
    
    Args:
        model: machine learning model with predict method.
        X_test: list of messages used for validation.
        Y_test: targets for comparing with model's predictions.
        category_names: list of category names in Y_test.
    """
    Y_pred = model.predict(X_test)

    scores = []
    precision = []
    recall = []

    for i in range(Y_test.shape[1]):
        scores.append(classification_report(Y_test[:, i], Y_pred[:, i], output_dict=True, zero_division=0)['weighted avg']['f1-score'])
        precision.append(classification_report(Y_test[:, i], Y_pred[:, i], output_dict=True, zero_division=0)['weighted avg']['precision'])
        recall.append(classification_report(Y_test[:, i], Y_pred[:, i], output_dict=True, zero_division=0)['weighted avg']['recall'])

    print("    f1 score: {:.2f}".format(sum(scores)/len(scores)))
    print("    precision: {:.2f}".format(sum(precision)/len(precision)))
    print("    recall: {:.2f}".format(sum(recall)/len(recall)))


def save_model(model, model_filepath):
    """
    Saves the model pickle in the filepath informed.
    
    Args:
        model: machine learning model.
        model_filepath: filepath where the model should be saved.
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


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