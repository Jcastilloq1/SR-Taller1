import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

ratings = pd.read_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/test.csv")
movies = pd.read_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/movie.csv")

matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)


# SIMILITUDES


def cosine_similarity_model():

    sim = cosine_similarity(matrix.fillna(0))

    return pd.DataFrame(sim,index=matrix.index,columns=matrix.index)


def pearson_similarity_model():

    return matrix.T.corr(method="pearson").fillna(0)


def jaccard_similarity_model():

    binary = matrix.notnull().astype(int)

    dist = pdist(binary,metric="jaccard")

    sim = 1 - squareform(dist)

    return pd.DataFrame(sim,index=binary.index,columns=binary.index)

# PREDICCION USUARIO-USUARIO


def predict_user_user(user_id,movie_id,sim_matrix,k=10):

    neighbors = sim_matrix[user_id].sort_values(ascending=False)

    neighbors = neighbors.iloc[1:k+1]

    ratings_neighbors = []
    weights = []

    for n in neighbors.index:

        r = matrix.loc[n,movie_id]

        if not np.isnan(r):

            ratings_neighbors.append(r)
            weights.append(neighbors[n])

    if len(ratings_neighbors)==0:
        return None

    ratings_neighbors=np.array(ratings_neighbors)
    weights=np.array(weights)

    pred = np.dot(ratings_neighbors,weights)/np.sum(np.abs(weights))

    return pred


# ITEM ITEM


def item_item_similarity():

    sim = cosine_similarity(matrix.fillna(0).T)

    return pd.DataFrame(sim,index=matrix.columns,columns=matrix.columns)


def recommend_item_item(user_id, n=5):

    if user_id not in matrix.index:
        return []

    sim = item_item_similarity()

    user_ratings = matrix.loc[user_id]

    rated_movies = user_ratings[user_ratings.notna()]

    predictions = []

    for candidate_movie in matrix.columns:

        if not pd.isna(user_ratings[candidate_movie]):
            continue

        numerator = 0
        denominator = 0

        for rated_movie, rating in rated_movies.items():

            similarity = sim.loc[candidate_movie, rated_movie]

            if similarity <= 0:
                continue

            numerator += similarity * rating
            denominator += abs(similarity)

        if denominator > 0:

            pred_rating = numerator / denominator

            predictions.append((candidate_movie, pred_rating))

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    results = []

    for movie_id, pred in predictions[:n]:

        title = movies[movies.movieId == movie_id].title.values[0]

        results.append({
            "movie": title,
            "pred_rating": round(pred, 2)
        })

    return results



# RECOMENDADOR GENERAL


def recommend(user_id,method="user",similarity="cosine",n=5):

    if method=="item":

        return recommend_item_item(user_id,n)

    if similarity=="cosine":
        sim=cosine_similarity_model()

    elif similarity=="pearson":
        sim=pearson_similarity_model()

    else:
        sim=jaccard_similarity_model()

    user_movies=matrix.loc[user_id]

    unseen=user_movies[user_movies.isna()].index

    preds=[]

    for movie in unseen:

        pred=predict_user_user(user_id,movie,sim)

        if pred is not None:

            preds.append((movie,pred))

    preds=sorted(preds,key=lambda x:x[1],reverse=True)[:n]

    results=[]

    for movie,pred in preds:

        title=movies[movies.movieId==movie].title.values[0]

        results.append({
            "movie":title,
            "pred_rating":round(pred,2)
        })

    return results