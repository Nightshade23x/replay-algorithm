from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from src.recommenders.online_logistic_mf import (
    OnlineLogisticMF,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:

    warmup_interactions: int = 5_000
    evaluation_interactions: int = 25_000

    candidate_pool_size: int = 50

    max_impressions_per_user_item: int = 3

    popularity_alpha: float = 1.0
    popularity_prior: float = 1.0

    latent_dim: int = 16
    learning_rate: float = 0.04
    regularization: float = 0.002

    top_k: int = 20


# ---------------------------------------------------------
# WARM-UP STATE
# ---------------------------------------------------------

@dataclass
class WarmupState:

    consumed: np.ndarray

    impressions: np.ndarray

    popularity_clicks: np.ndarray

    users: np.ndarray

    items: np.ndarray

    outcomes: np.ndarray


# ---------------------------------------------------------
# GINI
# ---------------------------------------------------------

def gini(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    if np.all(
        values == 0
    ):
        return 0.0

    values = np.sort(
        values
    )

    n = len(
        values
    )

    cumulative = np.cumsum(
        values
    )

    return (
        (n + 1)
        - (
            2
            * cumulative.sum()
            / cumulative[-1]
        )
    ) / n


# ---------------------------------------------------------
# ELIGIBILITY
# ---------------------------------------------------------

def eligible_items(
    user,
    consumed,
    impressions,
    config,
):
    """
    An item is eligible if:

    1. the user has not positively consumed it, and
    2. it has been shown fewer than the allowed number
       of times.

    This allows unsuccessful recommendations to reappear
    while preventing unlimited repeated impressions.
    """

    mask = (
        (~consumed[user])
        & (
            impressions[user]
            < config.max_impressions_per_user_item
        )
    )

    available = np.flatnonzero(
        mask
    )

    # Defensive fallback.
    if len(available) == 0:

        available = np.flatnonzero(
            ~consumed[user]
        )

    if len(available) == 0:

        available = np.arange(
            consumed.shape[1]
        )

    return available


# ---------------------------------------------------------
# RANDOM RECOMMENDER
# ---------------------------------------------------------

def random_recommend(
    rng,
    available,
):

    return int(
        rng.choice(
            available
        )
    )


# ---------------------------------------------------------
# PROPORTIONAL POPULARITY
# ---------------------------------------------------------

def proportional_popularity_recommend(
    rng,
    available,
    popularity_clicks,
    config,
):

    weights = (
        popularity_clicks[
            available
        ].astype(float)
        + config.popularity_prior
    ) ** config.popularity_alpha

    probabilities = (
        weights
        / weights.sum()
    )

    return int(
        rng.choice(
            available,
            p=probabilities,
        )
    )


# ---------------------------------------------------------
# GREEDY POPULARITY
# ---------------------------------------------------------

def greedy_popularity_recommend(
    rng,
    available,
    popularity_clicks,
):

    counts = (
        popularity_clicks[
            available
        ]
    )

    maximum = counts.max()

    candidates = available[
        counts == maximum
    ]

    return int(
        rng.choice(
            candidates
        )
    )


# ---------------------------------------------------------
# COLLABORATIVE FILTERING
# ---------------------------------------------------------

def build_cf_model(
    warmup,
    num_users,
    num_items,
    config,
):

    observed_click_rate = float(
        warmup.outcomes.mean()
    )

    observed_click_rate = float(
        np.clip(
            observed_click_rate,
            0.05,
            0.95,
        )
    )

    model = OnlineLogisticMF(
        num_users=num_users,
        num_items=num_items,
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
        regularization=config.regularization,
        initial_click_rate=observed_click_rate,
        seed=12345,
    )

    for user, item, outcome in zip(
        warmup.users,
        warmup.items,
        warmup.outcomes,
    ):

        model.update(
            int(user),
            int(item),
            int(outcome),
        )

    return model


def cf_recommend(
    model,
    rng,
    user,
    available,
    config,
):

    if (
        len(available)
        > config.candidate_pool_size
    ):

        candidates = rng.choice(
            available,
            size=config.candidate_pool_size,
            replace=False,
        )

    else:

        candidates = available

    scores = model.score_candidates(
        user,
        candidates,
    )

    maximum = scores.max()

    best_positions = np.flatnonzero(
        np.isclose(
            scores,
            maximum,
        )
    )

    selected_position = int(
        rng.choice(
            best_positions
        )
    )

    return int(
        candidates[
            selected_position
        ]
    )


# ---------------------------------------------------------
# CREATE IDENTICAL RANDOM WARM-UP
# ---------------------------------------------------------

def generate_warmup(
    click_probabilities,
    world_seed,
    config,
):

    num_users, num_items = (
        click_probabilities.shape
    )

    rng_users = np.random.default_rng(
        world_seed
    )

    rng_items = np.random.default_rng(
        world_seed + 1_000_000
    )

    rng_response = np.random.default_rng(
        world_seed + 2_000_000
    )

    consumed = np.zeros(
        (
            num_users,
            num_items,
        ),
        dtype=bool,
    )

    impressions = np.zeros(
        (
            num_users,
            num_items,
        ),
        dtype=np.int16,
    )

    popularity_clicks = np.zeros(
        num_items,
        dtype=np.int32,
    )

    users = np.empty(
        config.warmup_interactions,
        dtype=np.int32,
    )

    items = np.empty(
        config.warmup_interactions,
        dtype=np.int32,
    )

    outcomes = np.empty(
        config.warmup_interactions,
        dtype=np.int8,
    )

    for step in range(
        config.warmup_interactions
    ):

        user = int(
            rng_users.integers(
                num_users
            )
        )

        available = eligible_items(
            user,
            consumed,
            impressions,
            config,
        )

        item = random_recommend(
            rng_items,
            available,
        )

        probability = (
            click_probabilities[
                user,
                item
            ]
        )

        outcome = int(
            rng_response.random()
            < probability
        )

        impressions[
            user,
            item
        ] += 1

        if outcome == 1:

            consumed[
                user,
                item
            ] = True

            popularity_clicks[
                item
            ] += 1

        users[
            step
        ] = user

        items[
            step
        ] = item

        outcomes[
            step
        ] = outcome

    return WarmupState(
        consumed=consumed,
        impressions=impressions,
        popularity_clicks=popularity_clicks,
        users=users,
        items=items,
        outcomes=outcomes,
    )


# ---------------------------------------------------------
# SHARED EVALUATION RANDOMNESS
# ---------------------------------------------------------

def generate_evaluation_randomness(
    world_seed,
    num_users,
    config,
):
    """
    Every algorithm receives the same:

    - user-arrival sequence
    - response uniform random numbers
    - exploration decisions

    This makes algorithm comparisons much more controlled.
    """

    rng_users = np.random.default_rng(
        world_seed + 3_000_000
    )

    rng_response = np.random.default_rng(
        world_seed + 4_000_000
    )

    rng_exploration = np.random.default_rng(
        world_seed + 5_000_000
    )

    users = rng_users.integers(
        0,
        num_users,
        size=config.evaluation_interactions,
    )

    response_uniforms = (
        rng_response.random(
            config.evaluation_interactions
        )
    )

    exploration_uniforms = (
        rng_exploration.random(
            config.evaluation_interactions
        )
    )

    return (
        users,
        response_uniforms,
        exploration_uniforms,
    )


# ---------------------------------------------------------
# SIMULATE ONE ALGORITHM
# ---------------------------------------------------------

def simulate_algorithm(
    algorithm,
    epsilon,
    click_probabilities,
    underlying_item_relevance,
    warmup,
    evaluation_users,
    response_uniforms,
    exploration_uniforms,
    recommender_seed,
    config,
):

    num_users, num_items = (
        click_probabilities.shape
    )

    consumed = (
        warmup.consumed.copy()
    )

    impressions = (
        warmup.impressions.copy()
    )

    popularity_clicks = (
        warmup.popularity_clicks.copy()
    )

    evaluation_clicks = np.zeros(
        num_items,
        dtype=np.int32,
    )

    evaluation_exposures = np.zeros(
        num_items,
        dtype=np.int32,
    )

    rng_recommender = np.random.default_rng(
        recommender_seed
    )

    if algorithm == "cf":

        model = build_cf_model(
            warmup,
            num_users,
            num_items,
            config,
        )

    else:

        model = None

    total_clicks = 0

    total_hidden_relevance = 0.0

    exploration_count = 0

    for step in range(
        config.evaluation_interactions
    ):

        user = int(
            evaluation_users[
                step
            ]
        )

        available = eligible_items(
            user,
            consumed,
            impressions,
            config,
        )

        # -------------------------------------------------
        # RANDOM
        # -------------------------------------------------

        if algorithm == "random":

            item = random_recommend(
                rng_recommender,
                available,
            )

        # -------------------------------------------------
        # PROPORTIONAL POPULARITY
        # -------------------------------------------------

        elif (
            algorithm
            == "proportional_popularity"
        ):

            item = proportional_popularity_recommend(
                rng_recommender,
                available,
                popularity_clicks,
                config,
            )

        # -------------------------------------------------
        # GREEDY POPULARITY
        # -------------------------------------------------

        elif (
            algorithm
            == "greedy_popularity"
        ):

            item = greedy_popularity_recommend(
                rng_recommender,
                available,
                popularity_clicks,
            )

        # -------------------------------------------------
        # COLLABORATIVE FILTERING
        # -------------------------------------------------

        elif algorithm == "cf":

            explore = (
                exploration_uniforms[
                    step
                ]
                < epsilon
            )

            if explore:

                item = random_recommend(
                    rng_recommender,
                    available,
                )

                exploration_count += 1

            else:

                item = cf_recommend(
                    model,
                    rng_recommender,
                    user,
                    available,
                    config,
                )

        else:

            raise ValueError(
                f"Unknown algorithm: "
                f"{algorithm}"
            )

        # -------------------------------------------------
        # EXPOSURE
        # -------------------------------------------------

        impressions[
            user,
            item
        ] += 1

        evaluation_exposures[
            item
        ] += 1

        probability = (
            click_probabilities[
                user,
                item
            ]
        )

        total_hidden_relevance += (
            probability
        )

        # Common random number across algorithms.
        outcome = int(
            response_uniforms[
                step
            ]
            < probability
        )

        # -------------------------------------------------
        # POSITIVE INTERACTION
        # -------------------------------------------------

        if outcome == 1:

            evaluation_clicks[
                item
            ] += 1

            popularity_clicks[
                item
            ] += 1

            consumed[
                user,
                item
            ] = True

            total_clicks += 1

        # -------------------------------------------------
        # ONLINE LEARNING
        # -------------------------------------------------

        if model is not None:

            model.update(
                user,
                item,
                outcome,
            )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    ctr = (
        total_clicks
        / config.evaluation_interactions
    )

    shown_relevance = (
        total_hidden_relevance
        / config.evaluation_interactions
    )

    popularity_gini = gini(
        evaluation_clicks
    )

    exposure_gini = gini(
        evaluation_exposures
    )

    catalogue_coverage = (
        np.count_nonzero(
            evaluation_exposures
        )
        / num_items
    )

    relevance_correlation = (
        spearmanr(
            underlying_item_relevance,
            evaluation_clicks,
        ).statistic
    )

    if np.isnan(
        relevance_correlation
    ):

        relevance_correlation = 0.0

    underlying_top_k = set(
        np.argsort(
            underlying_item_relevance
        )[
            -config.top_k:
        ]
    )

    observed_top_k = set(
        np.argsort(
            evaluation_clicks
        )[
            -config.top_k:
        ]
    )

    top_k_relevance_recall = (
        len(
            underlying_top_k
            & observed_top_k
        )
        / config.top_k
    )

    if algorithm == "cf":

        actual_exploration_rate = (
            exploration_count
            / config.evaluation_interactions
        )

    else:

        actual_exploration_rate = 0.0

    return {
        "clicks":
            evaluation_clicks,

        "exposures":
            evaluation_exposures,

        "total_clicks":
            total_clicks,

        "ctr":
            ctr,

        "shown_relevance":
            shown_relevance,

        "popularity_gini":
            popularity_gini,

        "exposure_gini":
            exposure_gini,

        "catalogue_coverage":
            catalogue_coverage,

        "relevance_correlation":
            relevance_correlation,

        "top_k_relevance_recall":
            top_k_relevance_recall,

        "actual_exploration_rate":
            actual_exploration_rate,
    }