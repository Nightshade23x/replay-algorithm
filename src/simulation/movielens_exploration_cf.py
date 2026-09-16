import numpy as np

from src.recommenders.online_logistic_mf import (
    OnlineLogisticMF,
)

from src.simulation.movielens_online_cf import (
    load_preferences,
    preference_to_click_probability,
    random_recommender,
    evaluate_worlds,
    gini,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_WORLDS = 50

INTERACTIONS_PER_WORLD = 30_000

WARMUP_INTERACTIONS = 5_000

CANDIDATE_POOL_SIZE = 50

BASE_SEED = 2026

MODEL_SEED = 12345


# Probability of ignoring the CF recommendation
# and exploring randomly.
EPSILON_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
]


# ---------------------------------------------------------
# SIMULATE ONE WORLD
# ---------------------------------------------------------

def simulate_world(
    epsilon,
    world_seed,
    click_probabilities,
):

    num_users, num_items = (
        click_probabilities.shape
    )

    rng_users = np.random.default_rng(
        world_seed
    )

    rng_recommender = np.random.default_rng(
        world_seed + 1_000_000
    )

    rng_response = np.random.default_rng(
        world_seed + 2_000_000
    )

    seen = np.zeros(
        (
            num_users,
            num_items,
        ),
        dtype=bool,
    )

    click_counts = np.zeros(
        num_items,
        dtype=int,
    )

    exposure_counts = np.zeros(
        num_items,
        dtype=int,
    )

    model = OnlineLogisticMF(
        num_users=num_users,
        num_items=num_items,
        latent_dim=16,
        learning_rate=0.04,
        regularization=0.002,
        initial_click_rate=0.40,
        seed=MODEL_SEED,
    )

    total_clicks = 0

    total_oracle_probability = 0.0

    exploration_count = 0

    for step in range(
        INTERACTIONS_PER_WORLD
    ):

        user = rng_users.integers(
            num_users
        )

        # ---------------------------------------------
        # RANDOM WARM-UP
        # ---------------------------------------------

        if (
            step
            < WARMUP_INTERACTIONS
        ):

            item = random_recommender(
                rng_recommender,
                user,
                seen,
            )

        # ---------------------------------------------
        # EPSILON-GREEDY RECOMMENDATION
        # ---------------------------------------------

        else:

            explore = (
                rng_recommender.random()
                < epsilon
            )

            if explore:

                item = random_recommender(
                    rng_recommender,
                    user,
                    seen,
                )

                exploration_count += 1

            else:

                item = model.recommend(
                    rng_recommender,
                    user,
                    seen,
                    candidate_pool_size=(
                        CANDIDATE_POOL_SIZE
                    ),
                )

        # ---------------------------------------------
        # RECORD EXPOSURE
        # ---------------------------------------------

        exposure_counts[
            item
        ] += 1

        seen[
            user,
            item
        ] = True

        # ---------------------------------------------
        # SIMULATED USER RESPONSE
        # ---------------------------------------------

        probability = (
            click_probabilities[
                user,
                item
            ]
        )

        total_oracle_probability += (
            probability
        )

        clicked = (
            rng_response.random()
            < probability
        )

        outcome = int(
            clicked
        )

        if clicked:

            click_counts[
                item
            ] += 1

            total_clicks += 1

        # ---------------------------------------------
        # LEARN FROM OBSERVED FEEDBACK
        # ---------------------------------------------

        model.update(
            user,
            item,
            outcome,
        )

    recommendation_phase = (
        INTERACTIONS_PER_WORLD
        - WARMUP_INTERACTIONS
    )

    return {
        "clicks":
            click_counts,

        "exposures":
            exposure_counts,

        "total_clicks":
            total_clicks,

        "ctr":
            (
                total_clicks
                / INTERACTIONS_PER_WORLD
            ),

        "mean_oracle_probability":
            (
                total_oracle_probability
                / INTERACTIONS_PER_WORLD
            ),

        "click_gini":
            gini(
                click_counts
            ),

        "exposure_gini":
            gini(
                exposure_counts
            ),

        "actual_exploration_rate":
            (
                exploration_count
                / recommendation_phase
            ),
    }


# ---------------------------------------------------------
# RUN ONE EPSILON CONDITION
# ---------------------------------------------------------

def run_condition(
    epsilon,
    click_probabilities,
):

    results = []

    exploration_rates = []

    for world in range(
        NUM_WORLDS
    ):

        print(
            f"  epsilon={epsilon:.2f} "
            f"world {world + 1}/{NUM_WORLDS}",
            end="\r",
        )

        result = simulate_world(
            epsilon,
            BASE_SEED + world,
            click_probabilities,
        )

        results.append(
            result
        )

        exploration_rates.append(
            result[
                "actual_exploration_rate"
            ]
        )

    metrics = evaluate_worlds(
        results
    )

    metrics[
        "mean_exploration_rate"
    ] = np.mean(
        exploration_rates
    )

    return metrics


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    (
        preference_scores,
        user_ids,
        movie_ids,
    ) = load_preferences()

    click_probabilities = (
        preference_to_click_probability(
            preference_scores
        )
    )

    print(
        "============================================================"
    )

    print(
        "MOVIELENS EXPLORATION-AWARE COLLABORATIVE FILTERING"
    )

    print(
        "============================================================"
    )

    print(
        f"Users: {len(user_ids)}"
    )

    print(
        f"Items: {len(movie_ids)}"
    )

    print(
        f"Worlds per condition: "
        f"{NUM_WORLDS}"
    )

    print(
        f"Interactions per world: "
        f"{INTERACTIONS_PER_WORLD:,}"
    )

    print(
        f"Warm-up interactions: "
        f"{WARMUP_INTERACTIONS:,}"
    )

    print()

    print(
        "Epsilon | Rank corr | Top-20 | "
        "Pop Gini | Exp Gini | CTR | "
        "Shown relevance"
    )

    print(
        "-" * 79
    )

    for epsilon in (
        EPSILON_LEVELS
    ):

        metrics = run_condition(
            epsilon,
            click_probabilities,
        )

        print(
            " " * 80,
            end="\r",
        )

        print(
            f"{epsilon:>7.2f} | "
            f"{metrics['rank_correlation']:>9.3f} | "
            f"{metrics['top_k_overlap']:>6.3f} | "
            f"{metrics['click_gini']:>8.3f} | "
            f"{metrics['exposure_gini']:>8.3f} | "
            f"{metrics['mean_ctr']:>5.3f} | "
            f"{metrics['mean_oracle_probability']:>15.3f}"
        )

    print()

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()