import sys
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the data for messages and categories.
    The merge uses a left join from messages to categories on id column of both datasets.

    Args:
        messages_filepath: path for messages dataset.
        categories_filepath: path for categories dataset.
    Returns:
        dataframe: messages merged with categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages.merge(categories, how="left", on=["id"])

def clean_data(df):
    """
    Clean the data perfoming the following steps:
        1. Extract categories values and one hot encode them.
        2. Drop duplicates lines
    Args:
        df: dataframe
    Returns:
        dataframe: cleaned dataframed following the steps above. 
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    first_row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = first_row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df[categories.columns] = categories

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save the dataset into an sqlite database.
    Args:
        df: cleaned dataframe
        database_filename: path and name of sqlite database to be created;
    """
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql(database_filename, engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()