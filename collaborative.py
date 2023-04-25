# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load MovieLens dataset
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge datasets on movieId
data = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix
matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Compute cosine similarity between items
item_similarity = cosine_similarity(matrix.T)

# Define a function to make recommendations
def get_recommendations(movie_title, item_similarity):
    # Get the similarity scores for the movie
    similarity_scores = list(enumerate(item_similarity[movie_title]))
    # Sort movies based on similarity score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # Get top 10 most similar movies
    top_movies = [movie[0] for movie in similarity_scores[1:11]]
    return movies.iloc[top_movies]

# Get recommendations for a specific movie
get_recommendations('Toy Story (1995)', item_similarity)

