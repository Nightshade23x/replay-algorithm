from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.data.load_movielens import load_ratings


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_USERS = 500
NUM_ITEMS = 500

LATENT_DIM = 30
TEST_SIZE = 0.20

RANDOM_SEED = 42


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movielens_preferences.npz"
)


# ---------------------------------------------------------
# SELECT USERS AND ITEMS
# ---------------------------------------------------------

def select_subset(ratings):
    """
    Select active MovieLens users and frequently rated movies.

    This gives us a reasonably dense user-item matrix while
    keeping the simulation computationally manageable.
    """

    user_counts = (
        ratings["user_id"]
        .value_counts()
    )

    selected_users = (
        user_counts
        .head(NUM_USERS)
        .index
    )

    subset = ratings[
        ratings["user_id"].isin(
            selected_users
        )
    ].copy()

    movie_counts = (
        subset["movie_id"]
        .value_counts()
    )

    selected_movies = (
        movie_counts
        .head(NUM_ITEMS)
        .index
    )

    subset = subset[
        subset["movie_id"].isin(
            selected_movies
        )
    ].copy()

    return subset


# ---------------------------------------------------------
# MAP ORIGINAL IDS TO MATRIX INDICES
# ---------------------------------------------------------

def create_indices(ratings):

    user_ids = np.sort(
        ratings["user_id"].unique()
    )

    movie_ids = np.sort(
        ratings["movie_id"].unique()
    )

    user_to_index = {
        user_id: index
        for index, user_id
        in enumerate(user_ids)
    }

    movie_to_index = {
        movie_id: index
        for index, movie_id
        in enumerate(movie_ids)
    }

    ratings = ratings.copy()

    ratings["user_idx"] = (
        ratings["user_id"]
        .map(user_to_index)
    )

    ratings["movie_idx"] = (
        ratings["movie_id"]
        .map(movie_to_index)
    )

    return (
        ratings,
        user_ids,
        movie_ids,
    )


# ---------------------------------------------------------
# TRAIN MATRIX FACTORISATION MODEL
# ---------------------------------------------------------

def fit_model(
    train_ratings,
    num_users,
    num_items,
):

    global_mean = (
        train_ratings["rating"].mean()
    )

    user_means = (
        train_ratings
        .groupby("user_idx")["rating"]
        .mean()
    )

    user_mean_array = np.full(
        num_users,
        global_mean,
        dtype=float,
    )

    for user_idx, mean_rating in (
        user_means.items()
    ):

        user_mean_array[
            user_idx
        ] = mean_rating

    residuals = (
        train_ratings["rating"].to_numpy()
        - user_mean_array[
            train_ratings[
                "user_idx"
            ].to_numpy()
        ]
    )

    matrix = csr_matrix(
        (
            residuals,
            (
                train_ratings[
                    "user_idx"
                ].to_numpy(),
                train_ratings[
                    "movie_idx"
                ].to_numpy(),
            ),
        ),
        shape=(
            num_users,
            num_items,
        ),
    )

    svd = TruncatedSVD(
        n_components=LATENT_DIM,
        random_state=RANDOM_SEED,
    )

    user_factors = (
        svd.fit_transform(
            matrix
        )
    )

    item_factors = (
        svd.components_
    )

    reconstructed = (
        user_factors
        @ item_factors
    )

    predicted_ratings = (
        user_mean_array[:, None]
        + reconstructed
    )

    predicted_ratings = np.clip(
        predicted_ratings,
        1.0,
        5.0,
    )

    return (
        predicted_ratings,
        svd.explained_variance_ratio_.sum(),
    )


# ---------------------------------------------------------
# VALIDATE MODEL
# ---------------------------------------------------------

def validate_model(
    ratings,
    num_users,
    num_items,
):

    (
        train,
        test,
    ) = train_test_split(
        ratings,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    (
        predictions,
        explained_variance,
    ) = fit_model(
        train,
        num_users,
        num_items,
    )

    predicted_test = predictions[
        test["user_idx"].to_numpy(),
        test["movie_idx"].to_numpy(),
    ]

    actual_test = (
        test["rating"]
        .to_numpy()
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_test,
            predicted_test,
        )
    )

    mae = mean_absolute_error(
        actual_test,
        predicted_test,
    )

    return (
        rmse,
        mae,
        explained_variance,
    )


# ---------------------------------------------------------
# BUILD FINAL PREFERENCE MATRIX
# ---------------------------------------------------------

def build_preference_matrix(
    ratings,
    num_users,
    num_items,
):

    (
        predicted_ratings,
        explained_variance,
    ) = fit_model(
        ratings,
        num_users,
        num_items,
    )

    # Convert predicted ratings from [1, 5]
    # into a normalized preference score [0, 1].
    #
    # IMPORTANT:
    # This is a preference score, NOT yet a click probability.

    preference_scores = (
        predicted_ratings - 1.0
    ) / 4.0

    return (
        predicted_ratings,
        preference_scores,
        explained_variance,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "=========================================="
    )

    print(
        "BUILD MOVIELENS PREFERENCE MODEL"
    )

    print(
        "=========================================="
    )

    ratings = load_ratings()

    print(
        f"Original ratings: "
        f"{len(ratings):,}"
    )

    subset = select_subset(
        ratings
    )

    (
        subset,
        user_ids,
        movie_ids,
    ) = create_indices(
        subset
    )

    num_users = len(
        user_ids
    )

    num_items = len(
        movie_ids
    )

    possible_ratings = (
        num_users
        * num_items
    )

    density = (
        len(subset)
        / possible_ratings
    )

    print(
        f"Selected users: "
        f"{num_users}"
    )

    print(
        f"Selected movies: "
        f"{num_items}"
    )

    print(
        f"Observed ratings: "
        f"{len(subset):,}"
    )

    print(
        f"Matrix density: "
        f"{density:.2%}"
    )

    print()

    print(
        "Validating latent preference model..."
    )

    (
        rmse,
        mae,
        validation_variance,
    ) = validate_model(
        subset,
        num_users,
        num_items,
    )

    print(
        f"Validation RMSE: "
        f"{rmse:.3f}"
    )

    print(
        f"Validation MAE: "
        f"{mae:.3f}"
    )

    print(
        f"Validation explained variance: "
        f"{validation_variance:.3f}"
    )

    print()

    print(
        "Refitting using all selected ratings..."
    )

    (
        predicted_ratings,
        preference_scores,
        final_variance,
    ) = build_preference_matrix(
        subset,
        num_users,
        num_items,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH,
        user_ids=user_ids,
        movie_ids=movie_ids,
        predicted_ratings=(
            predicted_ratings.astype(
                np.float32
            )
        ),
        preference_scores=(
            preference_scores.astype(
                np.float32
            )
        ),
    )

    print(
        f"Final explained variance: "
        f"{final_variance:.3f}"
    )

    print()

    print(
        "Preference score range:"
    )

    print(
        f"Minimum: "
        f"{preference_scores.min():.3f}"
    )

    print(
        f"Maximum: "
        f"{preference_scores.max():.3f}"
    )

    print(
        f"Mean: "
        f"{preference_scores.mean():.3f}"
    )

    print()

    print(
        "Saved preference model to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()