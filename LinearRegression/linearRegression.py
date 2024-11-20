import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score # to check accuracy

# Step 2: we are going to load the CSV files now in the memory
# we are going to use two file, movies and ratings file for our linear regression. Our target is to get the ratings
# regressed line for each movie..

# Load the movies file and the ratings file
movies = pd.read_csv('ml-20m/movies.csv', sep=',')
ratings = pd.read_csv('ml-20m/ratings.csv', sep=',')

# Step 3: Preprocessing : we need to merge the two dataframes together on a column and make a sing dataframe
print(movies.columns) # to know the list of the column values present in the dataframe
print(ratings.columns) # to know the list of the column values present in the dataframe

finalDF = pd.merge(movies, ratings, on='movieId', how='inner')
print('Columns of full dataframe merged', finalDF.columns)

# drop the string columns, like title and genres from the dataframe and timestamp as well, because we are not going to use it
del finalDF['title']
del finalDF['genres']
del finalDF['timestamp']

# in addition to these, we should ensure the datatype of movieId, userId are integer and ratings are float
finalDF['movieId'] = finalDF['movieId'].astype(int)
finalDF['userId'] = finalDF['userId'].astype(int)
finalDF['rating'] = finalDF['rating'].astype(float)