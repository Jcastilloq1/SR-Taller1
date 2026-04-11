"""
Backend Flask para la interfaz de Lovable.
Sirve la API REST que el frontend espera.

Endpoints:
  GET  /api/users                    → User[]
  GET  /api/users/<id>               → User
  GET  /api/users/<id>/ratings       → { rating: Rating, movie: Movie }[]
  GET  /api/movies                   → Movie[]
  GET  /api/recommendations/<userId> → Recommendation[]
  POST /api/users                    → User
  POST /api/experiments/run          → ExperimentResult
  GET  /api/experiments/compare      → ExperimentResult[]
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Importar módulos del sistema de recomendación
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'processing'))

from recommender_system import (
    matrix, ratings, movies, user_means,
    cosine_similarity_model, pearson_similarity_model, jaccard_similarity_model,
    item_cosine_similarity, item_pearson_similarity, item_jaccard_similarity,
    predict_user_user, predict_item_item,
    get_user_ratings, get_all_user_ids,
)
from model_evaluator import evaluate_user_user, evaluate_item_item
from user_manager import add_new_user, get_next_user_id

app = Flask(__name__)
CORS(app)

# ============================================================
# USUARIOS NUEVOS (en memoria, para la demo)
# ============================================================
new_users = {}  # {user_id: {"name": str, "ratings": {movieId: rating, ...}}}

# ============================================================
# CACHE de matrices de similitud (disco + memoria)
# Se guardan como .pkl para no recalcular cada vez que se
# reinicia el servidor.
# ============================================================
CACHE_DIR = Path(__file__).parent.parent / "models"
CACHE_DIR.mkdir(exist_ok=True)

similarity_cache = {}


def get_similarity_matrix(model, model_type):
    """
    Obtiene la matriz de similitud:
    1. Si está en memoria (similarity_cache), la retorna directo.
    2. Si existe en disco (.pkl), la carga.
    3. Si no, la calcula, la guarda en disco y en memoria.
    """
    key = f"{model_type}_{model}"
    pkl_path = CACHE_DIR / f"{key}.pkl"

    # 1. En memoria
    if key in similarity_cache:
        return similarity_cache[key]

    # 2. En disco
    if pkl_path.exists():
        print(f"  Cargando {key} desde disco...")
        t0 = time.time()
        similarity_cache[key] = pd.read_pickle(pkl_path)
        print(f"  Listo: {time.time() - t0:.1f}s")
        return similarity_cache[key]

    # 3. Calcular y guardar
    print(f"  Calculando similitud: {key}...")
    t0 = time.time()
    if model_type == "user-user":
        if model == "coseno":
            similarity_cache[key] = cosine_similarity_model()
        elif model == "pearson":
            similarity_cache[key] = pearson_similarity_model()
        else:
            similarity_cache[key] = jaccard_similarity_model()
    else:
        if model == "coseno":
            similarity_cache[key] = item_cosine_similarity()
        elif model == "pearson":
            similarity_cache[key] = item_pearson_similarity()
        else:
            similarity_cache[key] = item_jaccard_similarity()

    elapsed = time.time() - t0
    print(f"  Calculado en {elapsed:.1f}s. Guardando en disco...")
    similarity_cache[key].to_pickle(pkl_path)
    print(f"  Guardado: {pkl_path.name}")

    return similarity_cache[key]


def precompute_all():
    """
    Pre-calcula todas las matrices de similitud al arrancar.
    Si ya están en disco, las carga (~1 segundo cada una).
    Si no, las calcula y guarda (minutos la primera vez).
    """
    print("\n  Pre-cargando matrices de similitud...")
    total_t0 = time.time()
    for model_type in ["item-item", "user-user"]:
        for model in ["coseno", "pearson", "jaccard"]:
            get_similarity_matrix(model, model_type)
    print(f"\n  Todas las matrices listas en {time.time() - total_t0:.1f}s\n")


# ============================================================
# HELPERS para formatear datos según el contrato del frontend
# ============================================================

def format_user(user_id):
    """Formatea un usuario según el contrato: { id, name, totalRatings, avgRating }"""
    # Nuevo usuario en memoria
    if user_id in new_users:
        u = new_users[user_id]
        r = list(u["ratings"].values())
        return {
            "id": int(user_id),
            "name": u["name"],
            "totalRatings": len(r),
            "avgRating": round(sum(r) / len(r), 1) if r else 0.0,
        }
    # Usuario existente en la matriz
    if user_id not in matrix.index:
        return None
    n_ratings = int(matrix.loc[user_id].notna().sum())
    avg = float(user_means[user_id]) if user_id in user_means.index else 0.0
    return {
        "id": int(user_id),
        "name": f"Usuario {user_id}",
        "totalRatings": n_ratings,
        "avgRating": round(avg, 1),
    }


def format_movie(movie_id):
    """Formatea una película según el contrato: { id, title, genres: string[], year }"""
    row = movies[movies.movieId == movie_id]
    if len(row) == 0:
        return {"id": int(movie_id), "title": f"Movie {movie_id}", "genres": [], "year": 0}
    row = row.iloc[0]
    title = row["title"]
    genres_str = row.get("genres", "")
    genres = genres_str.split("|") if genres_str and genres_str != "(no genres listed)" else []

    # Extraer año del título (formato: "Movie Title (1995)")
    year = 0
    match = re.search(r'\((\d{4})\)', title)
    if match:
        year = int(match.group(1))

    return {
        "id": int(movie_id),
        "title": title,
        "genres": genres,
        "year": year,
    }


def get_neighbors_for_prediction(user_id, movie_id, sim_matrix, model_type, k=30, threshold=0.0):
    """
    Calcula los vecinos usados para una predicción específica.
    Retorna lista de Neighbor según el contrato del frontend.
    """
    neighbors = []

    if model_type == "user-user":
        if user_id not in sim_matrix.index:
            return []
        sims = sim_matrix[user_id].sort_values(ascending=False)
        sims = sims.iloc[1:]  # excluir al usuario mismo

        for n_id in sims.index:
            s = sims[n_id]
            if s <= threshold:
                continue
            r = matrix.loc[n_id, movie_id] if movie_id in matrix.columns else np.nan
            if np.isnan(r):
                continue

            # Contar co-ratings
            corated = matrix.loc[user_id].notna() & matrix.loc[n_id].notna()

            neighbors.append({
                "id": int(n_id),
                "label": f"Usuario {n_id}",
                "similarity": round(float(s), 4),
                "commonItems": int(corated.sum()),
            })
            if len(neighbors) >= k:
                break
    else:
        # item-item: vecinos son películas similares que el usuario calificó
        if movie_id not in sim_matrix.index:
            return []
        user_ratings_series = matrix.loc[user_id] if user_id in matrix.index else None
        if user_ratings_series is None:
            return []

        sims = sim_matrix[movie_id].sort_values(ascending=False)

        for item_id in sims.index:
            if item_id == movie_id:
                continue
            s = sims[item_id]
            if s <= threshold:
                continue
            r = user_ratings_series[item_id] if item_id in user_ratings_series.index else np.nan
            if np.isnan(r):
                continue

            movie_info = format_movie(item_id)

            # Contar usuarios que calificaron ambos items
            corated = matrix[movie_id].notna() & matrix[item_id].notna()

            neighbors.append({
                "id": int(item_id),
                "label": movie_info["title"],
                "similarity": round(float(s), 4),
                "commonItems": int(corated.sum()),
            })
            if len(neighbors) >= k:
                break

    return neighbors


# ============================================================
# ENDPOINTS
# ============================================================

@app.route("/api/users", methods=["GET"])
def api_get_users():
    """GET /api/users → User[]"""
    user_ids = get_all_user_ids()
    users = [format_user(uid) for uid in user_ids]
    return jsonify(users)


@app.route("/api/users/<int:user_id>", methods=["GET"])
def api_get_user(user_id):
    """GET /api/users/:id → User"""
    user = format_user(user_id)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user)


@app.route("/api/users/<int:user_id>/ratings", methods=["GET"])
def api_get_user_ratings(user_id):
    """GET /api/users/:id/ratings → { rating: Rating, movie: Movie }[]"""
    # Nuevo usuario
    if user_id in new_users:
        result = []
        for mid, r in new_users[user_id]["ratings"].items():
            result.append({
                "rating": {"userId": user_id, "movieId": int(mid), "rating": r},
                "movie": format_movie(int(mid)),
            })
        return jsonify(sorted(result, key=lambda x: -x["rating"]["rating"]))

    # Usuario existente
    user_ratings_list = get_user_ratings(user_id)

    result = []
    for r in user_ratings_list:
        result.append({
            "rating": {
                "userId": user_id,
                "movieId": r["movieId"],
                "rating": r["rating"],
            },
            "movie": format_movie(r["movieId"]),
        })

    return jsonify(result)


@app.route("/api/movies", methods=["GET"])
def api_get_movies():
    """GET /api/movies → Movie[]"""
    # Retornar películas populares (las más calificadas)
    movie_counts = ratings.groupby("movieId").size().sort_values(ascending=False)
    top_movie_ids = movie_counts.head(100).index.tolist()

    result = [format_movie(mid) for mid in top_movie_ids]
    return jsonify(result)


@app.route("/api/recommendations/<int:user_id>", methods=["GET"])
def api_get_recommendations(user_id):
    """
    GET /api/recommendations/:userId → Recommendation[]
    Query params: model, modelType, k, threshold
    """
    model = request.args.get("model", "coseno")
    model_type = request.args.get("modelType", "item-item")
    k = int(request.args.get("k", 10))
    threshold = float(request.args.get("threshold", 0.0))

    # ---- Nuevo usuario: calcular recomendaciones con item-item directo ----
    if user_id in new_users:
        user_ratings = new_users[user_id]["ratings"]
        # Siempre usar item-item para nuevos usuarios (no están en la matriz user-user)
        sim_matrix = get_similarity_matrix(model, "item-item")

        rated_ids = set(user_ratings.keys())
        recommendations = []

        for movie_id in sim_matrix.index:
            if movie_id in rated_ids:
                continue

            # Calcular predicción manualmente
            numerator = 0.0
            denominator = 0.0
            neighbor_list = []

            for rated_id, rating in user_ratings.items():
                if rated_id not in sim_matrix.index:
                    continue
                s = sim_matrix.loc[movie_id, rated_id]
                if s > threshold:
                    numerator += s * rating
                    denominator += abs(s)
                    neighbor_list.append({
                        "id": int(rated_id),
                        "label": format_movie(int(rated_id))["title"],
                        "similarity": round(float(s), 4),
                        "commonItems": 0,
                    })

            if denominator > 0 and len(neighbor_list) >= 2:
                pred = numerator / denominator
                pred = float(np.clip(pred, 0.5, 5.0))
                neighbor_list.sort(key=lambda x: -x["similarity"])
                recommendations.append({
                    "movie": format_movie(movie_id),
                    "predictedRating": round(pred, 2),
                    "neighbors": neighbor_list[:k],
                })

        recommendations.sort(key=lambda x: -x["predictedRating"])
        return jsonify(recommendations[:20])

    # ---- Usuario existente: flujo normal ----
    if user_id not in matrix.index:
        return jsonify({"error": "User not found"}), 404

    # Obtener matriz de similitud
    sim_matrix = get_similarity_matrix(model, model_type)

    # Predecir para películas no vistas
    user_movies = matrix.loc[user_id]
    unseen = user_movies[user_movies.isna()].index

    recommendations = []
    for movie_id in unseen:
        if model_type == "user-user":
            pred = predict_user_user(user_id, movie_id, sim_matrix, k=k, min_neighbors=2)
        else:
            pred = predict_item_item(user_id, movie_id, sim_matrix, k=k, min_neighbors=2)

        if pred is not None:
            neighbors = get_neighbors_for_prediction(
                user_id, movie_id, sim_matrix, model_type, k=k, threshold=threshold
            )
            recommendations.append({
                "movie": format_movie(movie_id),
                "predictedRating": round(pred, 2),
                "neighbors": neighbors,
            })

    # Ordenar por rating predicho descendente y tomar top 20
    recommendations.sort(key=lambda x: -x["predictedRating"])
    recommendations = recommendations[:20]

    return jsonify(recommendations)


@app.route("/api/users", methods=["POST"])
def api_create_user():
    """POST /api/users → User"""
    data = request.json
    name = data.get("name", "Nuevo Usuario")
    ratings_list = data.get("ratings", [])

    if len(ratings_list) < 3:
        return jsonify({"error": "Se requieren al menos 3 ratings"}), 400

    new_id = get_next_user_id()

    # Guardar en memoria para recomendaciones inmediatas
    ratings_dict = {r["movieId"]: r["rating"] for r in ratings_list}
    new_users[new_id] = {"name": name, "ratings": ratings_dict}

    # También guardar en CSV para persistencia
    add_new_user(new_id, ratings_dict)

    avg = sum(r["rating"] for r in ratings_list) / len(ratings_list)

    print(f"  Nuevo usuario creado: id={new_id}, name={name}, ratings={len(ratings_list)}")

    return jsonify({
        "id": new_id,
        "name": name,
        "totalRatings": len(ratings_list),
        "avgRating": round(avg, 1),
    })


@app.route("/api/experiments/run", methods=["POST"])
def api_run_experiment():
    """POST /api/experiments/run → ExperimentResult"""
    params = request.json

    model = params.get("model", "coseno")
    model_type = params.get("modelType", "user-user")
    k = params.get("kNeighbors", 10)

    if model_type == "user-user":
        metrics = evaluate_user_user(model, max_rows=500)
    else:
        metrics = evaluate_item_item(model, max_rows=500)

    return jsonify({
        "params": params,
        "mae": metrics["MAE"] or 0,
        "rmse": metrics["RMSE"] or 0,
        "coverage": metrics.get("coverage", 0),
        "totalPredictions": metrics.get("n_predictions", 0),
    })


@app.route("/api/experiments/compare", methods=["GET"])
def api_compare_experiments():
    """GET /api/experiments/compare → ExperimentResult[]"""
    models_str = request.args.get("models", "jaccard,coseno,pearson")
    model_type = request.args.get("modelType", "user-user")
    k = int(request.args.get("k", 10))

    models_list = models_str.split(",")
    results = []

    for model in models_list:
        model = model.strip()
        print(f"  Evaluando {model_type} {model}...")

        if model_type == "user-user":
            metrics = evaluate_user_user(model, max_rows=300)
        else:
            metrics = evaluate_item_item(model, max_rows=300)

        results.append({
            "params": {
                "model": model,
                "modelType": model_type,
                "kNeighbors": k,
                "similarityThreshold": 0,
                "useMcLaughlin": False,
                "mcLaughlinBeta": 50,
            },
            "mae": metrics["MAE"] or 0,
            "rmse": metrics["RMSE"] or 0,
            "coverage": metrics.get("coverage", 0),
            "totalPredictions": metrics.get("n_predictions", 0),
        })

    return jsonify(results)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  Backend RecSys - Taller 1")
    print(f"  Usuarios: {matrix.shape[0]}")
    print(f"  Películas: {matrix.shape[1]}")
    print(f"  Ratings: {len(ratings)}")
    print("=" * 50)

    precompute_all()

    print("  Iniciando servidor en http://localhost:5000")
    print("  CORS habilitado para el frontend de Lovable\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
