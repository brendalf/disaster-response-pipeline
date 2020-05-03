# Disaster Response Pipeline
Analyzing message data for disaster response.
In this project, we developed a model to classify messages for disaster response.

## Project Structure
```
app -> dashboard source code.
data -> message dataset, categories dataset and etl pipeline source code.
models -> folder for saving models pickle file and ml pipeline source code.
notebooks -> notebooks for data analysis.
```  

## How to Run
Run the following commands in the project's root directory to set up your database and model.

### Data preparation
To run ETL pipeline that cleans data and stores in database

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### Training and saving model
To run ML pipeline that trains classifier and saves
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### Running dashboard server
Run the following command in the app's directory to run your web app.
`python app/run.py`

Go to http://0.0.0.0:3001/
