import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
''' 
The Surprise (Simple Python RecommendatIon System Engine) library is a Python library for building recommender systems. 
It provides a set of algorithms for collaborative filtering, as well as tools for evaluating and comparing the performance of different recommendation algorithms.
Built-in tools for performing matrix factorization algorithms like SVD (Singular Value Decomposition) and Non negative Matrix Factorization (NMF) as well as
for evaluation metrics like RMSE and MAE are available within it
'''

# Load data into pandas dataframe
df = pd.read_csv('ratings.csv')

# Define rating scale for surprise library
reader = Reader(rating_scale=(1, 5))  #rating scale can be changed, modified ormdeleted according to your use case
'''
The Reader object is used to specify the format of the input data for the surprise library. 
In this case, the rating_scale parameter is specifying the range of possible ratings in the input data.
Specifically, it is saying that the minimum rating value is 1 and the maximum rating value is 5.
'''


# Load data into surprise Dataset
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

# Split data into train and test sets
trainset, testset = train_test_split(data, test_size=.25)

# Train SVD algorithm on the training set
algo = SVD()
algo.fit(trainset)

# Generate recommendations for a given user
def get_recommendations(user_id, top_n=10):
    # Get all items the user has rated
    rated_items = df[df['user_id'] == user_id]['movie_id'].unique().tolist()

    # Get all items the user has not rated
    all_items = df['movie_id'].unique().tolist()
    unrated_items = [item for item in all_items if item not in rated_items]

    # Create a dataframe of unrated items for the user
    test_df = pd.DataFrame({'user_id': [user_id] * len(unrated_items),
                            'movie_id': unrated_items})

    # Convert dataframe to surprise Dataset and make predictions
    test_data = Dataset.load_from_df(test_df[['user_id', 'movie_id', 'rating']], reader)
    predictions = algo.test(test_data.build_full_trainset().build_testset())

    # Sort predictions by estimated rating
    pred_df = pd.DataFrame(predictions, columns=['user_id', 'movie_id', 'rating', 'est'])
    top_preds = pred_df[pred_df['movie_id'].isin(unrated_items)].sort_values('est', ascending=False).head(top_n)

    # Return top recommended items
    top_recs = top_preds['movie_id'].tolist()
    return top_recs

# Example usage:
user_id = 123
top_recs = get_recommendations(user_id, top_n=10)
print(top_recs)
