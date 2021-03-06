# Import Python libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    -read the messages file and the categories file, merge the two file into one

    Args:
    messages_file_path str: Messages CSV file
    categories_file_path str: Categories CSV file

    Returns:
    merged_df pandas_dataframe: merged dataframe of the two files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the messages and categories datasets using the common id
    df = messages.merge(categories, how = 'left', on='id')
    return df


def clean_data(df):
    '''
    Split the values in the categories column on the ; character so that each value becomes a separate column. 
    Use the first row of categories dataframe to create column names for the categories data.
    Rename columns of categories with new column names.

    Args:
    combined dataframe of messages and categories

    Returns:
    cleaned dataframe with no duplicates
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    categories.columns = [x[0] for x in categories.iloc[0].str.split("-")]
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    # convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].astype(str).str.split('-').apply(lambda x:x[1])
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace= True)
    return df

def save_data(df, database_filename):
    """
    Saves cleaned data to SQLite database

    Args:
    df pandas_dataframe: Cleaned data returned from clean_data() function
    database_file_name str: file name of the database to be saved to

    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    engine.dispose()


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