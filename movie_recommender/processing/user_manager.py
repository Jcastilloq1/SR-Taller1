import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RATINGS_FILE = DATA_DIR / "train.csv"


def add_rating(user_id, movie_id, rating):
    """
    Agrega un rating al archivo de entrenamiento.
    Permite que nuevos usuarios o nuevos ratings se incorporen al sistema.
    """
    ratings = pd.read_csv(RATINGS_FILE)

    # Verificar si ya existe este rating
    existing = ratings[(ratings["userId"] == user_id) & (ratings["movieId"] == movie_id)]

    if len(existing) > 0:
        # Actualizar rating existente
        ratings.loc[
            (ratings["userId"] == user_id) & (ratings["movieId"] == movie_id),
            "rating"
        ] = rating
        print(f"Rating actualizado: user={user_id}, movie={movie_id}, rating={rating}")
    else:
        # Agregar nuevo rating
        new_row = pd.DataFrame([{
            "userId": user_id,
            "movieId": movie_id,
            "rating": rating
        }])
        ratings = pd.concat([ratings, new_row], ignore_index=True)
        print(f"Rating agregado: user={user_id}, movie={movie_id}, rating={rating}")

    ratings.to_csv(RATINGS_FILE, index=False)


def add_new_user(user_id, ratings_dict):
    """
    Agrega un nuevo usuario con múltiples ratings.
    ratings_dict: {movieId: rating, ...}
    """
    for movie_id, rating in ratings_dict.items():
        add_rating(user_id, movie_id, rating)

    print(f"Usuario {user_id} creado con {len(ratings_dict)} ratings.")


def get_next_user_id():
    """Retorna el siguiente user ID disponible."""
    ratings = pd.read_csv(RATINGS_FILE)
    return int(ratings["userId"].max()) + 1
