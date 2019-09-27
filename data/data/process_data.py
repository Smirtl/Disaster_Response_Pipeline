import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads data from messages_filepath (csv) and categories_filepath (csv).
    Returns the merged dataframe, as well as the separate ones.
    '''
    messages = pd.read_csv(messages_filepath)
    categories1 = pd.read_csv(categories_filepath)
    df = messages.merge(categories1, on='id', how='left')

    return df, categories1, messages

def clean_data(df, categories1, messages):
    '''
    Does some data cleaning, such as splitting up the categories. Takes in all 3 files from the previous load_data function.
    Returns the cleaned dataframe.
    '''
    #Splitting the category column into 36 separate columns - one per category
    categories = categories1['categories'].str.split(pat=';', expand=True)
    categories.columns = [categories.iloc[0,i].split('-')[0] for i in range(36)]

    #Convert category values to just 1 or 0
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(-1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast = 'integer')

    #Concatenate categories with id column from categories1:
    categories = pd.concat([categories1['id'],categories] , axis=1)

    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = messages.merge(categories, on='id', how='left')

    print('Data merged.')

    #Remove duplicates
    a=df.shape[0]
    df.drop_duplicates(inplace=True)
    b=df.shape[0]
    print(str(a-b)+' duplicates dropped.')

    #Remove cases where the id is duplicated (with potentially contradicting values for the categories)
    a=df.shape[0]
    df.drop_duplicates(subset = 'id', keep = False, inplace = True)
    b=df.shape[0]
    print(str(a-b)+' rows dropped because their ids were not unique.')

    #Replace all the entries with value '2' for a category with the value '1' - it's assumed to be non-zero.
    a=df.shape[0]
    for column in df.columns[4:]:
        df[column] = df[column].replace(2,1)
    b=df.shape[0]
    print('Value "2" for a category replaced by "1" - to be regarded as non-zero.')

    #Return
    return df

def save_data(df, database_filename):
    '''
    Takes in the cleaned dataframe as well as a database filename of choice.
    Saves the data in an SQL databank under the chosen file name.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Categorized_Messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories1, messages = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories1, messages)

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
