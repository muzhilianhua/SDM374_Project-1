"""GUI smoke tests: every page renders without exceptions, with and without recorded runs."""
import pytest

pytest.importorskip("streamlit", reason="GUI tests need `uv sync --group gui`")
from streamlit.testing.v1 import AppTest  # noqa: E402

from sdm.cli import main  # noqa: E402
from sdm.paths import REPO_ROOT  # noqa: E402
from tests.conftest import BASELINE_CONFIG  # noqa: E402

PAGES = ("views/overview.py", "views/models.py", "views/experiments.py")


@pytest.fixture(scope="module")
def populated_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    for command in (["evaluate"], ["train"], ["optimize", "-n", "3"]):
        assert main(["--runs-dir", str(root), command[0], "-c", str(BASELINE_CONFIG), *command[1:]]) == 0
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


def test_overview_flags_extrapolating_baseline(populated_runs):
    app = render(populated_runs, PAGES[0])
    assert any("外推" in warning.value for warning in app.warning)
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["实测最低波动量"] == "115.76"
    assert labels["运行次数"] == "3"
