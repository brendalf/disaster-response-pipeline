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

pd.set_option('display.max_columns', 200)

nltk_download('punkt')
nltk_download('stopwords')
nltk_download('wordnet')


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    df = pd.read_sql_table('messages', engine, index_col="id")

    category_names = df.columns[4:]
    X = df.message.values
    Y = df[category_names].values
    return X, Y, category_names


def tokenize(text):
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
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())) 
    ])


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    scores = []
    for y_true, y_pred in zip(Y_test.T, Y_pred.T):
        scores.append(classification_report(y_true.T, y_pred.T, output_dict=True, zero_division=0)['weighted avg']['f1-score'])

    print("    f1 score: {:.2f}".format(sum(scores)/len(scores)))


def save_model(model, model_filepath):
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