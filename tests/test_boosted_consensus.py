"""The committee must expose an actual mean prediction and the intended minimax acquisition."""
import numpy as np

from sdm.models.boosted_consensus import BoostedConsensus


def test_acquisition_matches_worst_member_and_supports_single_weights():
    class Fixed:
        def __init__(self, mean, std):
            self.mean, self.std = mean, std

        def predict(self, X):
            return np.full(len(X), self.mean), np.full(len(X), self.std)

    model = BoostedConsensus()
    model.members = {"a": Fixed(100.0, 8.0), "b": Fixed(110.0, 3.0)}
    mean, envelope = model.predict(np.full((3, 7), 1 / 7))
    np.testing.assert_array_equal(mean, [105.0])
    np.testing.assert_array_equal(mean + envelope, [113.0])


def test_identical_members_keep_their_original_error_scale():
    class Fixed:
        def predict(self, X):
            return np.full(len(X), 100.0), np.full(len(X), 4.0)

    model = BoostedConsensus()
    model.members = {"a": Fixed(), "b": Fixed()}
    mean, envelope = model.predict(np.full((2, 21), 1 / 7))
    np.testing.assert_array_equal(mean, [100.0, 100.0])
    np.testing.assert_array_equal(envelope, [4.0, 4.0])
