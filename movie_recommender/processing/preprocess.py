import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess():

    print("Loading dataset...")

    ratings = pd.read_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/rating.csv")

    ratings = ratings.head(200000)

    train, test = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42
    )

    train.to_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/train.csv", index=False)
    test.to_csv("/Users/apple/Documents/Noveno Semestre/SisRec/Taller 1/SR-Taller1/movie_recommender/data/test.csv", index=False)

    print("Train/Test created")


if __name__ == "__main__":
    preprocess()