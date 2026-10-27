"""GUI tests: every page renders with and without runs, and buttons launch the right sdm commands."""
import sys

import pytest

pytest.importorskip("streamlit", reason="GUI tests need `uv sync --group gui`")
from streamlit.testing.v1 import AppTest  # noqa: E402

from sdm.cli import main  # noqa: E402
from sdm.paths import REPO_ROOT  # noqa: E402
from tests.conftest import BASELINE_CONFIG  # noqa: E402

PAGES = ("views/overview.py", "views/launch.py", "views/models.py", "views/experiments.py")

sys.path.insert(0, str(REPO_ROOT / "app"))
import jobs  # noqa: E402  (same module object the pages import)


class FakeJob:
    id = "fake-job"


@pytest.fixture
def started(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "start", lambda args, root, **kwargs: calls.append(list(args)) or FakeJob())
    return calls


@pytest.fixture(scope="module")
def populated_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    for command in (["evaluate"], ["train"], ["optimize", "-n", "3"]):
        assert main(["--runs-dir", str(root), command[0], "-c", str(BASELINE_CONFIG), *command[1:]]) == 0
    v32_config = REPO_ROOT / "experiments" / "baseline-v32.toml"
    assert main(["--runs-dir", str(root), "evaluate", "-c", str(v32_config)]) == 0
    return root


def render(runs_dir, page):
    app = AppTest.from_file(str(REPO_ROOT / "app" / "main.py"), default_timeout=30)
    app.session_state["runs_dir"] = str(runs_dir)
    app.run()
    if page != PAGES[0]:
        app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_runs(populated_runs, page):
    render(populated_runs, page)


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_without_runs(tmp_path, page):
    render(tmp_path / "empty", page)


def test_overview_leads_with_v32_score(populated_runs):
    app = render(populated_runs, PAGES[0])
    assert any(metric.label == "最佳 v3.2 总分" for metric in app.metric)


def test_overview_flags_extrapolating_baseline(populated_runs):
    app = render(populated_runs, PAGES[0])
    assert any("外推" in warning.value for warning in app.warning)
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["实测最低波动量"] == "115.76"
    assert labels["运行次数"] == "4"


def test_launch_button_runs_the_shown_command(populated_runs, started):
    app = render(populated_runs, "views/launch.py")
    app.button(key="launch_run").click().run()
    assert started == [["evaluate", "-c", "experiments/baseline.toml"]]


def test_reproduce_button_starts_a_reproduce_job(populated_runs, started):
    app = render(populated_runs, "views/experiments.py")
    chosen = app.selectbox(key="run_choice").value
    app.button(key="reproduce_run").click().run()
    assert started == [["reproduce", chosen]]


def test_predict_tab_scores_weights_with_the_saved_model(populated_runs):
    app = render(populated_runs, "views/experiments.py")
    train_run = next(run for run in app.selectbox(key="run_choice").options if "-train-" in run)
    app.selectbox(key="run_choice").set_value(train_run).run()
    app.button(key="predict_run").click().run()
    assert not app.exception, [e.value for e in app.exception]
    labels = {metric.label: metric.value for metric in app.metric}
    assert float(labels["预测波动量"]) > 0
