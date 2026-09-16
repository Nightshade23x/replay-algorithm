import numpy as np


class OnlineLogisticMF:
    """
    Online logistic matrix-factorisation recommender.

    The model learns from binary interaction feedback:

        clicked     -> 1
        exposed but not clicked -> 0

    It has NO access to the simulator's hidden preference
    probabilities.
    """

    def __init__(
        self,
        num_users,
        num_items,
        latent_dim=16,
        learning_rate=0.04,
        regularization=0.002,
        initial_click_rate=0.40,
        seed=12345,
    ):

        self.num_users = num_users
        self.num_items = num_items

        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization

        rng = np.random.default_rng(
            seed
        )

        # Small fixed random initialisation.
        self.user_factors = rng.normal(
            0.0,
            0.05,
            size=(
                num_users,
                latent_dim,
            ),
        )

        self.item_factors = rng.normal(
            0.0,
            0.05,
            size=(
                num_items,
                latent_dim,
            ),
        )

        self.user_bias = np.zeros(
            num_users,
            dtype=float,
        )

        self.item_bias = np.zeros(
            num_items,
            dtype=float,
        )

        # Start with a sensible global probability.
        self.global_bias = np.log(
            initial_click_rate
            / (
                1.0
                - initial_click_rate
            )
        )


    @staticmethod
    def _sigmoid(value):

        value = np.clip(
            value,
            -20.0,
            20.0,
        )

        return (
            1.0
            / (
                1.0
                + np.exp(
                    -value
                )
            )
        )


    def score_candidates(
        self,
        user,
        candidates,
    ):
        """
        Predict preference scores for candidate items.
        """

        latent_scores = (
            self.item_factors[
                candidates
            ]
            @ self.user_factors[
                user
            ]
        )

        scores = (
            self.global_bias
            + self.user_bias[
                user
            ]
            + self.item_bias[
                candidates
            ]
            + latent_scores
        )

        return scores


    def recommend(
        self,
        rng,
        user,
        seen,
        candidate_pool_size=50,
    ):
        """
        Recommend the highest-scoring item from a randomly
        sampled candidate pool of unseen items.
        """

        available = np.flatnonzero(
            ~seen[user]
        )

        if len(
            available
        ) == 0:

            available = np.arange(
                self.num_items
            )

        if (
            len(available)
            > candidate_pool_size
        ):

            candidates = rng.choice(
                available,
                size=candidate_pool_size,
                replace=False,
            )

        else:

            candidates = available

        scores = self.score_candidates(
            user,
            candidates,
        )

        max_score = scores.max()

        best_positions = np.flatnonzero(
            np.isclose(
                scores,
                max_score,
            )
        )

        selected_position = rng.choice(
            best_positions
        )

        return candidates[
            selected_position
        ]


    def update(
        self,
        user,
        item,
        outcome,
    ):
        """
        Perform one stochastic-gradient update after an
        observed exposure.

        outcome:
            1 -> clicked
            0 -> not clicked
        """

        user_vector = (
            self.user_factors[
                user
            ].copy()
        )

        item_vector = (
            self.item_factors[
                item
            ].copy()
        )

        score = (
            self.global_bias
            + self.user_bias[
                user
            ]
            + self.item_bias[
                item
            ]
            + np.dot(
                user_vector,
                item_vector,
            )
        )

        probability = (
            self._sigmoid(
                score
            )
        )

        error = (
            outcome
            - probability
        )

        lr = (
            self.learning_rate
        )

        reg = (
            self.regularization
        )

        self.user_factors[
            user
        ] += lr * (
            error
            * item_vector
            - reg
            * user_vector
        )

        self.item_factors[
            item
        ] += lr * (
            error
            * user_vector
            - reg
            * item_vector
        )

        self.user_bias[
            user
        ] += lr * (
            error
            - reg
            * self.user_bias[
                user
            ]
        )

        self.item_bias[
            item
        ] += lr * (
            error
            - reg
            * self.item_bias[
                item
            ]
        )

        # Update global click tendency more slowly.
        self.global_bias += (
            0.1
            * lr
            * error
        )