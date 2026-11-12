"""Integration checks for the recoverable T15/T16 policy adapter."""
import numpy as np

from sdm.interfaces import OnlineState
from sdm.registry import build_policy


def _state(dataset, *, call=1, X_online=None, y_online=None, online_call=None, failed=None,
           calls_left=None):
    X_online = np.empty((0, 21)) if X_online is None else np.asarray(X_online)
    y_online = np.empty(0) if y_online is None else np.asarray(y_online)
    online_call = np.empty(0, dtype=int) if online_call is None else np.asarray(online_call, dtype=int)
    failed = np.empty((0, 21)) if failed is None else np.asarray(failed)
    return OnlineState(call=call, calls_total=20, calls_left=20 if calls_left is None else calls_left,
                       batch_size=50, seed=2026, X_train=dataset.X, y_train=dataset.y,
                       X_online=X_online, y_online=y_online, online_call=online_call,
                       X_unmeasured=failed)


def _policy():
    from sdm.config import Component
    return build_policy("online-batch", {}, model=Component("ridge2", {}),
                        optimizer=Component("online-batch", {"pool_size": 700, "seed": 77}))


def test_policy_returns_oversupply_with_a_balanced_solver_prefix(dataset):
    proposal = _policy().propose(_state(dataset), 75)
    assert proposal.weights.shape == (75, 21)
    assert len(proposal.tags) == 75
    assert proposal.tags[:50].count("exploit") == 25
    assert proposal.tags[:50].count("explore") == 25
    assert len(np.unique(np.round(proposal.weights, 9), axis=0)) == 75


def test_policy_reconstructs_radius_and_excludes_failed_points(dataset):
    first = _policy().propose(_state(dataset), 75)
    measured = first.weights[:50]
    # An improved first batch expands the next trust region. A failed point is still excluded.
    state = _state(dataset, call=2, X_online=measured, y_online=np.linspace(100, 110, 50),
                   online_call=np.ones(50, dtype=int), failed=first.weights[50:55], calls_left=18)
    policy = _policy()
    proposal = policy.propose(state, 75)
    assert proposal.diagnostics["optimizer_radius_ratio"] == 1.5
    distances = np.abs(proposal.weights[:, None, :] - first.weights[50:55][None, :, :]).max(axis=2)
    assert distances.min() > 1e-8


def test_policy_is_identical_after_resume(dataset):
    state = _state(dataset)
    left = _policy().propose(state, 75)
    right = _policy().propose(state, 75)
    np.testing.assert_array_equal(left.weights, right.weights)
    assert left.tags == right.tags


def test_full_twenty_by_fifty_resume_matches_uninterrupted(tmp_path, monkeypatch):
    """Use the real T08 journal/log/recovery path with an analytic stand-in, never solver.exe."""
    import json
    from sdm.online import Evaluator
    from sdm.online import loop

    class Truth(Evaluator):
        kind = "test-truth"

        def _run(self, weights):
            # No improvements makes the radius shrink through its lower bound during the run.
            return 200 + np.square(weights).sum(axis=1), "", {}

    monkeypatch.setattr(loop, "build_evaluator", lambda *args, **kwargs: Truth())
    config = tmp_path / "policy.toml"
    config.write_text('''[model]
name = "ridge2"
[optimizer]
name = "online-batch"
[optimizer.params]
pool_size = 700
[policy]
name = "online-batch"
[evaluator]
kind = "synthetic"
[online]
calls = 20
batch_size = 50
candidate_margin = 25
''', encoding="utf-8")
    root = tmp_path / "runs"
    split, _ = loop.start(config, runs_root=root, name="split", max_calls=7, echo=lambda _: None)
    resumed, split_summary = loop.resume(split.id, runs_root=root, allow_code_change=True,
                                         echo=lambda _: None)
    straight, straight_summary = loop.start(config, runs_root=root, name="straight", echo=lambda _: None)
    a = loop.OnlineLog(resumed.path / "online_log.csv").rows()
    b = loop.OnlineLog(straight.path / "online_log.csv").rows()
    assert len(a) == len(b) == 1000
    assert split_summary["attempts"] == straight_summary["attempts"] == 20
    np.testing.assert_array_equal([row.weights for row in a], [row.weights for row in b])
    np.testing.assert_array_equal([row.R for row in a], [row.R for row in b])
    for call in range(1, 21):
        left = json.loads((resumed.path / "calls" / f"call_{call:02d}.json").read_text())
        right = json.loads((straight.path / "calls" / f"call_{call:02d}.json").read_text())
        assert left["tags"] == right["tags"]
        assert left["tags"].count("exploit") == left["tags"].count("explore") == 25
        assert left["diagnostics"]["state_radius_ratio"] == right["diagnostics"]["state_radius_ratio"]
        assert left["rejections"] == right["rejections"] == []
    assert left["diagnostics"]["state_radius_ratio"] == 0.25
