import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - (string) a valid filepath for the messages csv file
    categories_filepath - (string) a valid filepath for the categories csv file
    OUTPUT:
    df - (df) a merged dataframe of messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.concat([messages, categories], axis=1)
    
    return df


def clean_data(df):
    '''
    INPUT:
    df - (df) a dataframe of messages and their categories that needs to be cleaned
    OUTPUT:
    df - (df) a cleaned dataframe - creates dummy binary variables for each message category 
    '''
    print(df.shape)
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-').str[0]
    print(category_colnames)
    
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Loop over columns and remove text; convert values to  0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    
    #Check if any columns have an invalid value (should only be 1 or 0)
    cols = ((categories.max() > 1) | (categories.min() < 0))
    #Get the names of invalid columns
    col_indexes = cols[cols].index

    #Print column names of invalid columns
    print(col_indexes)

    rows_with_invalid_related_rows = categories.related > 1
    categories.loc[rows_with_invalid_related_rows, 'related'] = 1

    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

    pass


def save_data(df, database_filename):
    '''
    INPUT:
    df - (df) the dataframe to be saved as a database table called "messages"
    database_filename - (string) a valid filepath to save the database to
    
    Saves the dataframe in a SQL database referenced by database_filename
    '''
    print('save_data function exceuting...')
    print(df.shape)
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')
    pass  


def main():
    '''
    Load, clean and save the data
    '''
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