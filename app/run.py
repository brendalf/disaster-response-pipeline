import re
import json
import joblib
import plotly
import operator
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # number of messages per genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # proportion of messages per category
    category_names = df.columns[4:]
    category_perc = df[category_names].sum().values / df.shape[0]
    
    # most used words in death category
    messages = df[df['death'] == 1]['message'].values
    words = dict()
    for message in messages:
        for w in tokenize(message):
            try:
                words[w] += 1
            except:
                words[w] = 1

    words_sorted = dict(sorted(words.items(), key=operator.itemgetter(1), reverse=True))

    idx = 0
    topk = dict()
    for key, val in words_sorted.items():
        topk[key] = val
        idx += 1
        if idx == 5:
            break

    words = list(topk.keys())
    words_counts = list(topk.values())        

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, {
            'data': [
                Bar(
                    x=category_names,
                    y=category_perc
                )
            ],

            'layout': {
                'title': 'Distribution Percentage of Message Categories',
                'yaxis': {
                    'title': "Percentage",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'automargin':True
                }
            }
        }, {
            'data': [
                Bar(
                    x=words,
                    y=words_counts
                )
            ],

            'layout': {
                'title': 'Frequence of Most Used Words in Death Category',
                'yaxis': {
                    'title': "Count",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Words",
                    'automargin':True
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()