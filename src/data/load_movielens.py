from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ml-1m"
)


def load_ratings():

    ratings_path = DATA_DIR / "ratings.dat"

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp",
        ],
    )

    return ratings


def load_movies():

    movies_path = DATA_DIR / "movies.dat"

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=[
            "movie_id",
            "title",
            "genres",
        ],
        encoding="latin-1",
    )

    return movies


def load_users():

    users_path = DATA_DIR / "users.dat"

    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "gender",
            "age",
            "occupation",
            "zip_code",
        ],
    )

    return users


def main():

    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    print("==============================")
    print("MOVIELENS 1M VALIDATION")
    print("==============================")

    print(f"Ratings: {len(ratings):,}")
    print(f"Users in ratings: {ratings['user_id'].nunique():,}")
    print(f"Users in users.dat: {len(users):,}")
    print(f"Movies rated: {ratings['movie_id'].nunique():,}")
    print(f"Movies in catalogue: {len(movies):,}")

    print("\nRating distribution:")
    print(
        ratings["rating"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMissing values in ratings:")
    print(
        ratings.isna()
        .sum()
        .to_string()
    )

    print("\nDuplicate rating rows:")
    print(
        ratings.duplicated(
            subset=[
                "user_id",
                "movie_id",
                "timestamp",
            ]
        ).sum()
    )

    print("==============================")


if __name__ == "__main__":
    main()