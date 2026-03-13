import pandas as pd

ratings_file = "/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/train.csv"


def add_rating(user_id, movie_id, rating):

    ratings = pd.read_csv(ratings_file)

    new_row = pd.DataFrame([{
        "userId": user_id,
        "movieId": movie_id,
        "rating": rating
    }])

    ratings = pd.concat([ratings, new_row])

    ratings.to_csv(ratings_file, index=False)

    print("Rating saved")