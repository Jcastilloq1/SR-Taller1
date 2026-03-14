"""
==============================================================================
Preprocesamiento de datos - Taller 1 Modelos Colaborativos
==============================================================================

Este script toma el dataset MovieLens 20M y genera los archivos de datos
necesarios para el sistema de recomendación:

    movie_recommender/data/train.csv   (userId, movieId, rating)
    movie_recommender/data/test.csv    (userId, movieId, rating)
    movie_recommender/data/movie.csv   (movieId, title, genres)

Uso:
    python data_preprocessing.py --ml_path /ruta/a/ml-20m

    donde /ruta/a/ml-20m es la carpeta descomprimida del dataset MovieLens 20M
    descargado de https://grouplens.org/datasets/movielens/20m/

==============================================================================
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================
RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).parent / "movie_recommender" / "data"

# Parámetros de muestreo
SAMPLE_N_USERS = 1500          # Número de usuarios a muestrear
MIN_RATINGS_PER_USER = 20      # Mínimo de ratings por usuario
MIN_RATINGS_PER_MOVIE = 15     # Mínimo de ratings por película
TEST_RATIO = 0.2               # Proporción para test

np.random.seed(RANDOM_SEED)


# ============================================================
# PASO 1: Carga del dataset original
# ============================================================
def load_movielens(ml_path):
    """
    Carga ratings.csv y movies.csv del dataset MovieLens 20M.
    
    Estructura del dataset original:
        ratings.csv: userId, movieId, rating, timestamp
            - 20,000,263 ratings
            - 138,493 usuarios
            - 27,278 películas
            - Ratings de 0.5 a 5.0 en incrementos de 0.5
            - Timestamps en formato Unix epoch
        
        movies.csv: movieId, title, genres
            - Títulos incluyen año entre paréntesis
            - Géneros separados por "|"
    """
    ml_path = Path(ml_path)
    
    print("=" * 70)
    print("PASO 1: Carga del dataset MovieLens 20M")
    print("=" * 70)
    
    ratings_file = ml_path / "ratings.csv"
    movies_file = ml_path / "movies.csv"
    
    if not ratings_file.exists():
        raise FileNotFoundError(
            f"No se encontró {ratings_file}\n"
            f"Asegúrese de descomprimir ml-20m.zip y pasar la ruta correcta."
        )
    
    print(f"\nCargando {ratings_file}...")
    ratings_df = pd.read_csv(ratings_file)
    print(f"  Ratings totales:   {len(ratings_df):>12,}")
    print(f"  Usuarios únicos:   {ratings_df['userId'].nunique():>12,}")
    print(f"  Películas únicas:  {ratings_df['movieId'].nunique():>12,}")
    
    print(f"\nCargando {movies_file}...")
    movies_df = pd.read_csv(movies_file)
    print(f"  Películas en catálogo: {len(movies_df):,}")
    
    print(f"\nDistribución de ratings (original):")
    dist = ratings_df["rating"].value_counts().sort_index()
    for val, count in dist.items():
        pct = count / len(ratings_df) * 100
        bar = "█" * int(pct * 2)
        print(f"  {val:>4.1f}: {bar} {count:>10,} ({pct:.1f}%)")
    
    return ratings_df, movies_df


# ============================================================
# PASO 2: Muestreo estratégico
# ============================================================
def sample_dataset(ratings_df, movies_df):
    """
    Estrategia de muestreo para reducir MovieLens 20M a un tamaño
    manejable para experimentación.
    
    Justificación de la estrategia:
    ---------------------------------------------------------------
    1. Filtrar usuarios con >= MIN_RATINGS_PER_USER ratings:
       Usuarios con muy pocos ratings no generan perfiles confiables
       para collaborative filtering. Con 20+ ratings, hay suficiente
       información para calcular similitudes significativas.
    
    2. Muestrear SAMPLE_N_USERS usuarios aleatorios:
       La matriz user-user tiene complejidad O(n²). Con 1500 usuarios,
       la matriz es 1500x1500 = 2.25M celdas, completamente manejable.
       El muestreo es uniforme para no sesgar la representatividad.
    
    3. Filtrar películas con >= MIN_RATINGS_PER_MOVIE ratings en la muestra:
       Elimina películas con muy pocos ratings que producirían:
       - Similitudes item-item poco confiables
       - Sparsity innecesaria en la matriz
       - Overfitting en las predicciones
    
    4. Re-filtrar usuarios que perdieron ratings por el filtro de películas:
       Garantiza que todos los usuarios mantengan perfiles suficientes
       después de eliminar películas poco populares.
    
    Por qué muestrear por usuario (no por ratings aleatorios):
    ---------------------------------------------------------------
    El muestreo por usuario preserva los perfiles completos de cada
    usuario seleccionado. Un muestreo aleatorio de ratings destruiría
    los perfiles, haciendo imposible calcular similitudes user-user
    confiables. Es el estándar en evaluación de sistemas de recomendación.
    """
    print("\n" + "=" * 70)
    print("PASO 2: Muestreo estratégico")
    print("=" * 70)
    
    # 2.1 Filtrar usuarios activos
    user_counts = ratings_df.groupby("userId").size()
    active_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    print(f"\n  Usuarios con >= {MIN_RATINGS_PER_USER} ratings: "
          f"{len(active_users):,} / {ratings_df['userId'].nunique():,}")
    
    # 2.2 Muestrear usuarios
    if len(active_users) > SAMPLE_N_USERS:
        sampled_users = np.random.choice(
            active_users, size=SAMPLE_N_USERS, replace=False
        )
        print(f"  Usuarios muestreados: {len(sampled_users):,}")
    else:
        sampled_users = active_users.values
        print(f"  Usando todos los usuarios activos: {len(sampled_users):,}")
    
    sample_df = ratings_df[ratings_df["userId"].isin(sampled_users)].copy()
    print(f"  Ratings después de muestreo de usuarios: {len(sample_df):,}")
    
    # 2.3 Filtrar películas con suficientes ratings en la muestra
    movie_counts = sample_df.groupby("movieId").size()
    active_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
    sample_df = sample_df[sample_df["movieId"].isin(active_movies)]
    print(f"  Películas con >= {MIN_RATINGS_PER_MOVIE} ratings en muestra: "
          f"{sample_df['movieId'].nunique():,}")
    
    # 2.4 Re-filtrar usuarios
    user_counts_2 = sample_df.groupby("userId").size()
    valid_users = user_counts_2[user_counts_2 >= MIN_RATINGS_PER_USER].index
    sample_df = sample_df[sample_df["userId"].isin(valid_users)]
    
    # 2.5 Re-indexar userIds a 1..N para consistencia
    unique_users = sorted(sample_df["userId"].unique())
    user_map = {old: new for new, old in enumerate(unique_users, start=1)}
    sample_df["userId"] = sample_df["userId"].map(user_map)
    
    # Filtrar movies_df
    sample_movies = movies_df[
        movies_df["movieId"].isin(sample_df["movieId"].unique())
    ].copy()
    
    # Estadísticas finales
    n_users = sample_df["userId"].nunique()
    n_movies = sample_df["movieId"].nunique()
    n_ratings = len(sample_df)
    sparsity = 1 - n_ratings / (n_users * n_movies)
    
    print(f"\n  --- Muestra final ---")
    print(f"  Usuarios:   {n_users:,}")
    print(f"  Películas:  {n_movies:,}")
    print(f"  Ratings:    {n_ratings:,}")
    print(f"  Sparsity:   {sparsity:.4f} ({sparsity*100:.1f}%)")
    print(f"  Ratings/usuario:  min={sample_df.groupby('userId').size().min()}, "
          f"max={sample_df.groupby('userId').size().max()}, "
          f"media={sample_df.groupby('userId').size().mean():.1f}")
    print(f"  Ratings/película: min={sample_df.groupby('movieId').size().min()}, "
          f"max={sample_df.groupby('movieId').size().max()}, "
          f"media={sample_df.groupby('movieId').size().mean():.1f}")
    
    return sample_df, sample_movies


# ============================================================
# PASO 3: Conversión a formato compatible
# ============================================================
def convert_to_compatible_format(sample_df):
    """
    Transformación de datos para compatibilidad con modelos colaborativos.
    
    Decisiones tomadas:
    ---------------------------------------------------------------
    1. Se MANTIENEN los ratings explícitos (0.5 - 5.0):
       Los modelos user-user y item-item basados en Coseno y Pearson
       trabajan directamente con ratings numéricos. No se binarizan
       los datos porque perderíamos información valiosa sobre la
       intensidad de la preferencia.
    
    2. Para Jaccard, la binarización se hace DENTRO del modelo:
       Jaccard requiere datos binarios (le gustó / no le gustó).
       La binarización se aplica con umbral rating >= 3.5 → 1, else → 0.
       Esto se maneja en el cálculo de Jaccard, no en el preprocesamiento,
       para no perder los ratings originales que usan Coseno y Pearson.
    
    3. Se ELIMINA la columna timestamp para los archivos de datos:
       El timestamp se usa solo para el split temporal (ver Paso 4).
       Los modelos colaborativos no lo necesitan, y el código existente
       del sistema espera solo (userId, movieId, rating).
    
    4. Los datos de interacción usuario-ítem son los ratings explícitos:
       En MovieLens, el rating ES la interacción. No hay datos implícitos
       (clicks, views) que requieran transformación adicional.
    """
    print("\n" + "=" * 70)
    print("PASO 3: Conversión a formato compatible")
    print("=" * 70)
    
    print(f"\n  Columnas originales: {list(sample_df.columns)}")
    print(f"  Tipo de datos:")
    print(f"    userId:  {sample_df['userId'].dtype} (entero)")
    print(f"    movieId: {sample_df['movieId'].dtype} (entero)")
    print(f"    rating:  {sample_df['rating'].dtype} (flotante 0.5-5.0)")
    
    print(f"\n  Distribución de ratings en la muestra:")
    dist = sample_df["rating"].value_counts().sort_index()
    for val, count in dist.items():
        pct = count / len(sample_df) * 100
        bar = "█" * int(pct * 2)
        print(f"    {val:>4.1f}: {bar} {count:>8,} ({pct:.1f}%)")
    
    print(f"\n  Rating promedio: {sample_df['rating'].mean():.3f}")
    print(f"  Rating mediana:  {sample_df['rating'].median():.1f}")
    print(f"  Desv. estándar:  {sample_df['rating'].std():.3f}")
    
    # Los datos ya están en formato compatible (userId, movieId, rating)
    # Solo necesitamos el split (paso 4)
    
    return sample_df


# ============================================================
# PASO 4: Split Train / Test
# ============================================================
def split_train_test(sample_df):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Estrategia: Split temporal por usuario
    ---------------------------------------------------------------
    Para cada usuario, se ordenan sus ratings por timestamp (más 
    antiguos primero). El primer 80% de los ratings van a train y 
    el último 20% va a test.
    
    Justificación:
    - Es más realista que un split aleatorio: en producción, el sistema
      predice ratings FUTUROS basándose en el historial PASADO.
    - Evita data leakage temporal: no usamos información futura para
      predecir el pasado.
    - Es el estándar en la literatura de evaluación de RecSys.
    - Cada usuario tiene ratings tanto en train como en test, lo que
      permite evaluar las predicciones para todos los usuarios.
    
    Alternativa considerada y descartada:
    - Split aleatorio global: más simple pero no respeta la temporalidad.
      Podría usar un rating de 2020 para predecir uno de 2015.
    - Leave-one-out: deja solo 1 rating por usuario en test, 
      insuficiente para calcular métricas robustas.
    """
    print("\n" + "=" * 70)
    print("PASO 4: Split Train / Test (temporal)")
    print("=" * 70)
    
    train_list = []
    test_list = []
    
    for user_id, user_ratings in sample_df.groupby("userId"):
        user_ratings = user_ratings.sort_values("timestamp")
        n = len(user_ratings)
        
        split_idx = int(n * (1 - TEST_RATIO))
        # Asegurar al menos 1 en test y suficientes en train
        split_idx = max(5, min(split_idx, n - 1))
        
        train_list.append(user_ratings.iloc[:split_idx])
        test_list.append(user_ratings.iloc[split_idx:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    # Eliminar timestamp para los archivos finales
    train_out = train_df[["userId", "movieId", "rating"]].copy()
    test_out = test_df[["userId", "movieId", "rating"]].copy()
    
    # Verificar que todos los usuarios de test están en train
    test_only = set(test_out["userId"]) - set(train_out["userId"])
    if test_only:
        print(f"  WARN: {len(test_only)} usuarios solo en test, removiendo...")
        test_out = test_out[~test_out["userId"].isin(test_only)]
    
    print(f"\n  Train:")
    print(f"    Ratings:    {len(train_out):,}")
    print(f"    Usuarios:   {train_out['userId'].nunique():,}")
    print(f"    Películas:  {train_out['movieId'].nunique():,}")
    
    print(f"\n  Test:")
    print(f"    Ratings:    {len(test_out):,}")
    print(f"    Usuarios:   {test_out['userId'].nunique():,}")
    print(f"    Películas:  {test_out['movieId'].nunique():,}")
    
    # Verificar que no hay leakage
    print(f"\n  Verificaciones:")
    print(f"    Usuarios de test en train: "
          f"{len(set(test_out['userId']) & set(train_out['userId']))}"
          f" / {test_out['userId'].nunique()} ✓")
    
    return train_out, test_out


# ============================================================
# PASO 5: Guardar archivos
# ============================================================
def save_outputs(train_df, test_df, movies_df):
    """
    Guarda los archivos en el formato esperado por el sistema.
    
    Archivos generados:
        train.csv  - Datos de entrenamiento (userId, movieId, rating)
        test.csv   - Datos de evaluación (userId, movieId, rating)
        movie.csv  - Catálogo de películas (movieId, title, genres)
    """
    print("\n" + "=" * 70)
    print("PASO 5: Guardando archivos")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"
    movie_path = OUTPUT_DIR / "movie.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    movies_df.to_csv(movie_path, index=False)
    
    print(f"\n  {train_path}")
    print(f"    -> {len(train_df):,} filas, columnas: {list(train_df.columns)}")
    print(f"  {test_path}")
    print(f"    -> {len(test_df):,} filas, columnas: {list(test_df.columns)}")
    print(f"  {movie_path}")
    print(f"    -> {len(movies_df):,} filas, columnas: {list(movies_df.columns)}")
    
    # Tamaños de archivo
    for p in [train_path, test_path, movie_path]:
        size = os.path.getsize(p) / 1024
        unit = "KB"
        if size > 1024:
            size /= 1024
            unit = "MB"
        print(f"    Tamaño: {size:.1f} {unit}")


# ============================================================
# MAIN
# ============================================================
def main():
    global SAMPLE_N_USERS, MIN_RATINGS_PER_USER, MIN_RATINGS_PER_MOVIE
    global TEST_RATIO, RANDOM_SEED
    
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de MovieLens 20M para Taller 1"
    )
    parser.add_argument(
        "--ml_path",
        type=str,
        required=True,
        help="Ruta a la carpeta ml-20m descomprimida (contiene ratings.csv y movies.csv)"
    )
    parser.add_argument(
        "--n_users",
        type=int,
        default=SAMPLE_N_USERS,
        help=f"Número de usuarios a muestrear (default: {SAMPLE_N_USERS})"
    )
    parser.add_argument(
        "--min_user_ratings",
        type=int,
        default=MIN_RATINGS_PER_USER,
        help=f"Mínimo de ratings por usuario (default: {MIN_RATINGS_PER_USER})"
    )
    parser.add_argument(
        "--min_movie_ratings",
        type=int,
        default=MIN_RATINGS_PER_MOVIE,
        help=f"Mínimo de ratings por película (default: {MIN_RATINGS_PER_MOVIE})"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=TEST_RATIO,
        help=f"Proporción de datos para test (default: {TEST_RATIO})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Semilla aleatoria (default: {RANDOM_SEED})"
    )
    args = parser.parse_args()
    
    # Actualizar globales con argumentos
    SAMPLE_N_USERS = args.n_users
    MIN_RATINGS_PER_USER = args.min_user_ratings
    MIN_RATINGS_PER_MOVIE = args.min_movie_ratings
    TEST_RATIO = args.test_ratio
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    
    print("\n" + "=" * 70)
    print("  PREPROCESAMIENTO - Taller 1 Modelos Colaborativos")
    print("  Dataset: MovieLens 20M")
    print("=" * 70)
    print(f"\n  Parámetros:")
    print(f"    Ruta dataset:       {args.ml_path}")
    print(f"    Usuarios a samplear: {SAMPLE_N_USERS}")
    print(f"    Min ratings/usuario: {MIN_RATINGS_PER_USER}")
    print(f"    Min ratings/movie:   {MIN_RATINGS_PER_MOVIE}")
    print(f"    Test ratio:          {TEST_RATIO}")
    print(f"    Seed:                {RANDOM_SEED}")
    print(f"    Output dir:          {OUTPUT_DIR}")
    
    # Pipeline
    ratings_df, movies_df = load_movielens(args.ml_path)
    sample_df, sample_movies = sample_dataset(ratings_df, movies_df)
    sample_df = convert_to_compatible_format(sample_df)
    train_df, test_df = split_train_test(sample_df)
    save_outputs(train_df, test_df, sample_movies)
    
    print("\n" + "=" * 70)
    print("  PREPROCESAMIENTO COMPLETO")
    print("=" * 70)
    print(f"\n  Para ejecutar el sistema:")
    print(f"    cd movie_recommender/web")
    print(f"    python app.py")
    print()


if __name__ == "__main__":
    main()
