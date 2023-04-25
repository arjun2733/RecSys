# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load MovieLens dataset
movies = pd.read_csv('movies.csv')

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Preprocess movie genres
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Compute TF-IDF vectors for movie genres
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity between movies
movie_similarity = cosine_similarity(tfidf_matrix)

# Define a function to make recommendations
def get_recommendations(movie_title, movie_similarity, movies):
    # Get the similarity scores for the movie
    similarity_scores = list(enumerate(movie_similarity[movies[movies['title'] == movie_title].index[0]]))
    # Sort movies based on similarity score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # Get top 10 most similar movies
    top_movies = [movies.iloc[movie[0]]['title'] for movie in similarity_scores[1:11]]
    return top_movies

# Get recommendations for a specific movie
get_recommendations('Toy Story (1995)', movie_similarity, movies)
