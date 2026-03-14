import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Importar funciones del sistema de recomendación
from recommender_system import (
    matrix,
    predict_user_user,
    predict_item_item,
    cosine_similarity_model,
    pearson_similarity_model,
    jaccard_similarity_model,
    item_cosine_similarity,
    item_pearson_similarity,
    item_jaccard_similarity,
)

test = pd.read_csv(DATA_DIR / "test.csv")


# ============================================================
# EVALUACIÓN USER-USER
# ============================================================

def evaluate_user_user(similarity="cosine", max_rows=None):
    """
    Evalúa el modelo user-user sobre el conjunto de test.

    Retorna dict con MAE, RMSE, y número de predicciones realizadas.
    max_rows: limitar evaluación a N filas (None = todas).
    """
    if similarity == "cosine":
        sim = cosine_similarity_model()
    elif similarity == "pearson":
        sim = pearson_similarity_model()
    else:
        sim = jaccard_similarity_model()

    preds = []
    real = []

    eval_data = test if max_rows is None else test.head(max_rows)

    for _, row in eval_data.iterrows():
        user = row.userId
        movie = row.movieId

        if user not in matrix.index or movie not in matrix.columns:
            continue

        pred = predict_user_user(user, movie, sim)

        if pred is not None:
            preds.append(pred)
            real.append(row.rating)

    if len(preds) == 0:
        return {"MAE": None, "RMSE": None, "n_predictions": 0, "n_total": len(eval_data)}

    preds = np.array(preds)
    real = np.array(real)
    errors = real - preds

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "n_predictions": len(preds),
        "n_total": len(eval_data),
        "coverage": round(len(preds) / len(eval_data), 4),
    }


# ============================================================
# EVALUACIÓN ITEM-ITEM
# ============================================================

def evaluate_item_item(similarity="cosine", max_rows=None):
    """
    Evalúa el modelo item-item sobre el conjunto de test.
    """
    if similarity == "cosine":
        sim = item_cosine_similarity()
    elif similarity == "pearson":
        sim = item_pearson_similarity()
    else:
        sim = item_jaccard_similarity()

    preds = []
    real = []

    eval_data = test if max_rows is None else test.head(max_rows)

    for _, row in eval_data.iterrows():
        user = row.userId
        movie = row.movieId

        if user not in matrix.index or movie not in sim.index:
            continue

        pred = predict_item_item(user, movie, sim)

        if pred is not None:
            preds.append(pred)
            real.append(row.rating)

    if len(preds) == 0:
        return {"MAE": None, "RMSE": None, "n_predictions": 0, "n_total": len(eval_data)}

    preds = np.array(preds)
    real = np.array(real)
    errors = real - preds

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "n_predictions": len(preds),
        "n_total": len(eval_data),
        "coverage": round(len(preds) / len(eval_data), 4),
    }


# ============================================================
# EVALUACIÓN COMPLETA
# ============================================================

def get_all_metrics(max_rows=500):
    """
    Ejecuta evaluación para todos los modelos y similitudes.
    max_rows: limita filas de test para velocidad (500 por defecto).
    """
    results = {}

    for similarity in ["cosine", "pearson", "jaccard"]:
        print(f"  Evaluando user-user {similarity}...")
        results[f"user_{similarity}"] = evaluate_user_user(similarity, max_rows)

    for similarity in ["cosine", "pearson", "jaccard"]:
        print(f"  Evaluando item-item {similarity}...")
        results[f"item_{similarity}"] = evaluate_item_item(similarity, max_rows)

    return results


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    print("Evaluando modelos...")
    results = get_all_metrics(max_rows=500)

    print("\n" + "=" * 65)
    print(f"{'Modelo':<20} {'MAE':>8} {'RMSE':>8} {'Coverage':>10} {'N preds':>8}")
    print("=" * 65)

    for name, metrics in results.items():
        mae = f"{metrics['MAE']:.4f}" if metrics['MAE'] else "N/A"
        rmse = f"{metrics['RMSE']:.4f}" if metrics['RMSE'] else "N/A"
        cov = f"{metrics['coverage']:.2%}" if metrics.get('coverage') else "N/A"
        n = metrics['n_predictions']
        print(f"{name:<20} {mae:>8} {rmse:>8} {cov:>10} {n:>8}")
