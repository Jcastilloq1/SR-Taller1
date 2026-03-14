import pandas as pd
from surprise import Dataset, Reader, KNNBasic

ratings = pd.read_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/train.csv")
movies = pd.read_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/movie.csv")

reader = Reader(rating_scale=(0.5,5))

dataset = Dataset.load_from_df(
    ratings[['userId','movieId','rating']],
    reader
)

trainset = dataset.build_full_trainset()


# --------------------------------------------------
# CREAR MODELO
# --------------------------------------------------

def build_model(similarity="cosine", user_based=True):

    sim_map = {
        "cosine":"cosine",
        "pearson":"pearson",
        "jaccard":"msd"
    }

    sim_options = {
        "name": sim_map[similarity],
        "user_based": user_based
    }

    algo = KNNBasic(
        k=20,
        min_k=2,
        sim_options=sim_options
    )

    algo.fit(trainset)

    return algo


# --------------------------------------------------
# RECOMENDACIONES
# --------------------------------------------------

def recommend(user_id, method="item", similarity="cosine", n=5):

    user_based = True if method=="user" else False

    algo = build_model(similarity, user_based)

    user_movies = ratings[ratings.userId == user_id].movieId.values
    all_movies = movies.movieId.values

    unseen = [m for m in all_movies if m not in user_movies]

    predictions = []

    for movie in unseen:

        pred = algo.predict(user_id, movie)

        predictions.append((movie, pred.est))

    predictions = sorted(predictions,key=lambda x:x[1],reverse=True)[:n]

    results=[]

    for movie_id,pred in predictions:

        title = movies[movies.movieId==movie_id].title.values[0]

        results.append({
            "title":title,
            "pred_rating":round(pred,2)
        })

    return results





