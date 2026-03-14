import pandas as pd
from processing.recommender import recommend

ratings = pd.read_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/train.csv")
movies = pd.read_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/movie.csv")


def get_user_history(user_id):

    user_data = ratings[ratings.userId == user_id]

    history = user_data.merge(movies, on="movieId")

    return history[["title","rating"]]


def get_recommendations(user_id, method, similarity):

    recs = recommend(user_id, method, similarity)

    return pd.DataFrame(recs)