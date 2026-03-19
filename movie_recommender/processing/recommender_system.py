import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pathlib import Path

# ============================================================
# CARGA DE DATOS (rutas relativas)
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"

ratings = pd.read_csv(DATA_DIR / "train.csv")
movies = pd.read_csv(DATA_DIR / "movie.csv")

matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Pre-calcular medias por usuario (se usan en predicción y Pearson)
user_means = matrix.mean(axis=1)


# ============================================================
# SIMILITUDES USUARIO-USUARIO
# ============================================================

def cosine_similarity_model():
    """
    Coseno calculado SOLO sobre co-ratings.
    No usa fillna(0) porque eso trata 'no vio' como 'rating 0',
    inflando similitudes con coincidencias falsas en matrices sparse.
    """
    n = matrix.shape[0]
    users = matrix.index
    vals = matrix.values
    mask = ~np.isnan(vals)

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            corated = mask[i] & mask[j]
            if corated.sum() < 2:
                continue
            ri = vals[i, corated]
            rj = vals[j, corated]
            norm_i = np.linalg.norm(ri)
            norm_j = np.linalg.norm(rj)
            if norm_i > 0 and norm_j > 0:
                s = np.dot(ri, rj) / (norm_i * norm_j)
                sim[i, j] = s
                sim[j, i] = s

    return pd.DataFrame(sim, index=users, columns=users)


def pearson_similarity_model():
    """
    Correlación de Pearson sobre co-ratings.
    Centra los ratings de cada usuario por su media antes de calcular,
    lo que captura si las desviaciones correlacionan (no los valores absolutos).
    """
    n = matrix.shape[0]
    users = matrix.index
    vals = matrix.values
    mask = ~np.isnan(vals)
    means = user_means.values

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            corated = mask[i] & mask[j]
            if corated.sum() < 2:
                continue
            ri = vals[i, corated] - means[i]
            rj = vals[j, corated] - means[j]
            norm_i = np.linalg.norm(ri)
            norm_j = np.linalg.norm(rj)
            if norm_i > 0 and norm_j > 0:
                s = np.dot(ri, rj) / (norm_i * norm_j)
                sim[i, j] = s
                sim[j, i] = s

    return pd.DataFrame(sim, index=users, columns=users)


def jaccard_similarity_model():
    """
    Jaccard sobre presencia/ausencia de rating.
    Mide qué proporción de películas vistas son compartidas.
    """
    binary = matrix.notnull().astype(int)
    dist = pdist(binary, metric="jaccard")
    sim = 1 - squareform(dist)
    return pd.DataFrame(sim, index=binary.index, columns=binary.index)


# ============================================================
# SIMILITUDES ITEM-ITEM
# ============================================================

def item_cosine_similarity():
    """Coseno item-item sobre co-ratings (usuarios que calificaron ambos items)."""
    n = matrix.shape[1]
    items = matrix.columns
    vals = matrix.values  # users x items
    mask = ~np.isnan(vals)

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            corated = mask[:, i] & mask[:, j]
            if corated.sum() < 2:
                continue
            ri = vals[corated, i]
            rj = vals[corated, j]
            norm_i = np.linalg.norm(ri)
            norm_j = np.linalg.norm(rj)
            if norm_i > 0 and norm_j > 0:
                s = np.dot(ri, rj) / (norm_i * norm_j)
                sim[i, j] = s
                sim[j, i] = s

    return pd.DataFrame(sim, index=items, columns=items)


def item_pearson_similarity():
    """Pearson item-item: centra por media del item antes de correlacionar."""
    n = matrix.shape[1]
    items = matrix.columns
    vals = matrix.values
    mask = ~np.isnan(vals)
    item_means_vals = np.nanmean(vals, axis=0)

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            corated = mask[:, i] & mask[:, j]
            if corated.sum() < 2:
                continue
            ri = vals[corated, i] - item_means_vals[i]
            rj = vals[corated, j] - item_means_vals[j]
            norm_i = np.linalg.norm(ri)
            norm_j = np.linalg.norm(rj)
            if norm_i > 0 and norm_j > 0:
                s = np.dot(ri, rj) / (norm_i * norm_j)
                sim[i, j] = s
                sim[j, i] = s

    return pd.DataFrame(sim, index=items, columns=items)


def item_jaccard_similarity():
    """Jaccard item-item: proporción de usuarios compartidos."""
    binary = matrix.notnull().astype(int)
    dist = pdist(binary.T, metric="jaccard")
    sim = 1 - squareform(dist)
    return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)


# ============================================================
# PREDICCIÓN USUARIO-USUARIO (con mean centering)
# ============================================================

def predict_user_user(user_id, movie_id, sim_matrix, k=30, min_neighbors=2):
    """
    Predicción con mean centering:
        pred(u, i) = μ_u + Σ[sim(u,v) * (r(v,i) - μ_v)] / Σ|sim(u,v)|

    - k: número máximo de vecinos a considerar
    - min_neighbors: mínimo de vecinos que hayan calificado el item
      para generar predicción (evita predicciones basadas en 1 solo dato)
    """
    if user_id not in sim_matrix.index:
        return None
    if movie_id not in matrix.columns:
        return None

    neighbors = sim_matrix[user_id].sort_values(ascending=False)
    neighbors = neighbors.iloc[1:k + 1]

    mean_u = user_means[user_id]

    centered_ratings = []
    weights = []

    for n_id in neighbors.index:
        r = matrix.loc[n_id, movie_id]
        if not np.isnan(r):
            mean_n = user_means[n_id]
            centered_ratings.append(r - mean_n)
            weights.append(neighbors[n_id])

    if len(centered_ratings) < min_neighbors:
        return None

    centered_ratings = np.array(centered_ratings)
    weights = np.array(weights)

    denom = np.sum(np.abs(weights))
    if denom == 0:
        return None

    pred = mean_u + np.dot(centered_ratings, weights) / denom
    return float(np.clip(pred, 0.5, 5.0))


# ============================================================
# PREDICCIÓN ITEM-ITEM
# ============================================================

def predict_item_item(user_id, movie_id, sim_matrix, k=30, min_neighbors=2):
    """
    Predicción item-item:
        pred(u, i) = Σ[sim(i,j) * r(u,j)] / Σ|sim(i,j)|

    Solo considera items que el usuario ha calificado y que tienen
    similitud positiva con el item objetivo.
    """
    if user_id not in matrix.index:
        return None
    if movie_id not in sim_matrix.index:
        return None

    user_ratings = matrix.loc[user_id]
    rated_items = user_ratings[user_ratings.notna()]

    sims_and_ratings = []
    for rated_id, rating in rated_items.items():
        if rated_id == movie_id:
            continue
        if rated_id not in sim_matrix.columns:
            continue
        s = sim_matrix.loc[movie_id, rated_id]
        if s > 0:
            sims_and_ratings.append((s, rating))

    if len(sims_and_ratings) < min_neighbors:
        return None

    # Tomar los top-k más similares
    sims_and_ratings.sort(key=lambda x: -x[0])
    sims_and_ratings = sims_and_ratings[:k]

    numerator = sum(s * r for s, r in sims_and_ratings)
    denominator = sum(abs(s) for s, _ in sims_and_ratings)

    if denominator == 0:
        return None

    pred = numerator / denominator
    return float(np.clip(pred, 0.5, 5.0))


# ============================================================
# RECOMENDADOR USUARIO-USUARIO
# ============================================================

def recommend_user_user(user_id, sim_matrix, similarity_name="cosine", n=10):
    """Genera top-N recomendaciones usando modelo user-user."""
    if user_id not in matrix.index:
        return []

    user_movies = matrix.loc[user_id]
    unseen = user_movies[user_movies.isna()].index

    preds = []
    for movie in unseen:
        pred = predict_user_user(user_id, movie, sim_matrix)
        if pred is not None:
            preds.append((movie, pred))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:n]

    results = []
    for movie_id, pred in preds:
        title_match = movies[movies.movieId == movie_id].title.values
        title = title_match[0] if len(title_match) > 0 else f"Movie {movie_id}"
        genres_match = movies[movies.movieId == movie_id].genres.values
        genres = genres_match[0] if len(genres_match) > 0 else ""
        results.append({
            "movieId": int(movie_id),
            "movie": title,
            "genres": genres,
            "pred_rating": round(pred, 2)
        })

    return results


# ============================================================
# RECOMENDADOR ITEM-ITEM
# ============================================================

def recommend_item_item(user_id, sim_matrix, n=10):
    """Genera top-N recomendaciones usando modelo item-item."""
    if user_id not in matrix.index:
        return []

    user_movies = matrix.loc[user_id]
    unseen = user_movies[user_movies.isna()].index

    preds = []
    for movie in unseen:
        pred = predict_item_item(user_id, movie, sim_matrix)
        if pred is not None:
            preds.append((movie, pred))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:n]

    results = []
    for movie_id, pred in preds:
        title_match = movies[movies.movieId == movie_id].title.values
        title = title_match[0] if len(title_match) > 0 else f"Movie {movie_id}"
        genres_match = movies[movies.movieId == movie_id].genres.values
        genres = genres_match[0] if len(genres_match) > 0 else ""
        results.append({
            "movieId": int(movie_id),
            "movie": title,
            "genres": genres,
            "pred_rating": round(pred, 2)
        })

    return results


# ============================================================
# FUNCIÓN GENERAL DE RECOMENDACIÓN
# ============================================================

def recommend(user_id, method="user", similarity="cosine", n=10):
    """
    Punto de entrada principal.
    method: "user" (user-user) o "item" (item-item)
    similarity: "cosine", "pearson", "jaccard"
    """
    if user_id not in matrix.index:
        return []

    if method == "item":
        if similarity == "cosine":
            sim = item_cosine_similarity()
        elif similarity == "pearson":
            sim = item_pearson_similarity()
        else:
            sim = item_jaccard_similarity()
        return recommend_item_item(user_id, sim, n)
    else:
        if similarity == "cosine":
            sim = cosine_similarity_model()
        elif similarity == "pearson":
            sim = pearson_similarity_model()
        else:
            sim = jaccard_similarity_model()
        return recommend_user_user(user_id, sim, similarity, n)


# ============================================================
# UTILIDADES (para la app web y evaluación)
# ============================================================

def get_user_ratings(user_id):
    """Retorna los ratings del usuario con info de película."""
    if user_id not in matrix.index:
        return []

    user_data = ratings[ratings["userId"] == user_id]
    results = []
    for _, row in user_data.iterrows():
        mid = row["movieId"]
        title_match = movies[movies.movieId == mid].title.values
        title = title_match[0] if len(title_match) > 0 else f"Movie {mid}"
        genres_match = movies[movies.movieId == mid].genres.values
        genres = genres_match[0] if len(genres_match) > 0 else ""
        results.append({
            "movieId": int(mid),
            "movie": title,
            "genres": genres,
            "rating": row["rating"]
        })

    return sorted(results, key=lambda x: -x["rating"])


def get_all_user_ids():
    """Retorna lista de todos los user IDs disponibles."""
    return sorted(matrix.index.tolist())
