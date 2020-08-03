# Disaster Response Pipeline Project

## Summary
An application which loads a csv of messages about a disaster happening somewhere in the world (such as a flood, earthquake or famine).
The messages need to be appropriately classified into subjects, i.e. flags indicating whether the message relates to aid, a flood etc.
The application will load and clean the messages, spliting the message text into words using nltyk and lemmatization.
It will then save it as a database table called "messages".  The application then runs a GridSearchCV pipeline on the messages, training a
model on a split of the data, then evaluates the model classification performance on the data not used in training.

## Data
Two files are required
1. messages csv file - a csv file containing the messages to be classified
2. categories csv file - a csv file containing the categorisation of the supplied messages

## Web App
The web app allows the user to type/pase in a new message and click submit.  The model will then process the text of the
message and predict the relevant categories and highlight these on the web page.

The web app also shows three visualisations of the total data:
*  Distribution of message genres bar chart
*  Distribution of the top 10 most numerous message categories
*  Correlation matrix heatmap of each message category's correlation to each other category

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
