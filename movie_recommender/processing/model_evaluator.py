import numpy as np
from sklearn.metrics import mean_squared_error
from recommender_system import *

test=pd.read_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/test.csv")

def compute_rmse(similarity):

    preds=[]
    real=[]

    sim=None

    if similarity=="cosine":
        sim=cosine_similarity_model()

    elif similarity=="pearson":
        sim=pearson_similarity_model()

    else:
        sim=jaccard_similarity_model()

    for _,row in test.iterrows():

        user=row.userId
        movie=row.movieId
        rating=row.rating

        if user not in matrix.index or movie not in matrix.columns:
            continue

        pred=predict_user_user(user,movie,sim)

        if pred:

            preds.append(pred)
            real.append(rating)

    rmse=np.sqrt(mean_squared_error(real,preds))

    return round(rmse,3)


def get_all_rmse():

    return {
        "cosine":compute_rmse("cosine"),
        "pearson":compute_rmse("pearson"),
        "jaccard":compute_rmse("jaccard")
    }