"""T11 acceptance tests with append-only CSV fixtures; no evaluator is invoked."""
import json

import pandas as pd
import pytest

st = pytest.importorskip("streamlit", reason="GUI tests need the gui dependency group")
pa = pytest.importorskip("pyarrow")
from streamlit.testing.v1 import AppTest  # noqa: E402

from sdm.paths import REPO_ROOT  # noqa: E402
from sdm.simplex import FEATURES  # noqa: E402


def render(root):
    app = AppTest.from_file(str(REPO_ROOT / "app" / "main.py"), default_timeout=30)
    app.session_state["runs_dir"] = str(root)
    app.run()
    app.switch_page("views/online.py").run()
    assert not app.exception, [error.value for error in app.exception]
    return app


def labels(app):
    return {metric.label: metric.value for metric in app.metric}


def chart_values(app):
    # Streamlit renamed this element; inspect the data actually sent to the chart in either version.
    charts = app.get("vega_lite_chart") or app.get("arrow_vega_lite_chart")
    assert len(charts) == 1
    proto = charts[0].proto
    name = json.loads(proto.spec)["data"]["name"]
    dataset = next(dataset for dataset in proto.datasets if dataset.name == name)
    return pa.ipc.open_stream(dataset.data.data).read_pandas()


def test_online_console_shows_quota_checks_curve_and_latest_weights(tmp_path):
    frame = pd.DataFrame({
        "call": [1, 1, 2, 3], "R": [120.0, 119.0, 121.0, 118.0],
        "candidate_id": ["c1-1", "c1-2", "c2-1", "c3-1"],
        "strategy": ["exploit"] * 4, "run_id": ["t11-acceptance"] * 4,
        "timestamp": ["2026-11-12T16:00:00+08:00"] * 4,
        "simplex_ok": [True] * 4, "check_unique": [1.0] * 4,
        **{feature: [1 / 7] * 4 for feature in FEATURES},
    })
    frame.to_csv(tmp_path / "online_log.csv", index=False)

    app = render(tmp_path)

    assert labels(app) == {"已用调用额度": "3 / 20", "剩余调用额度": "17",
                           "已记录权重组": "4", "历史最低 R": "118.0000"}
    assert app.dataframe[0].value["全部通过"].tolist() == ["通过"] * 3
    curve = chart_values(app)
    assert curve["call"].tolist() == [1, 2, 3]
    assert curve["batch_best"].tolist() == [119.0, 121.0, 118.0]
    assert curve["best_so_far"].tolist() == [119.0, 119.0, 118.0]
    latest = app.dataframe[1].value
    assert latest["candidate_id"].tolist() == ["c3-1"]
    assert set(FEATURES).issubset(latest.columns)
    assert not app.error


def test_online_console_reports_failed_and_unknown_checks(tmp_path):
    pd.DataFrame({"call": [1, 2, 3], "R": [120.0, 119.0, 118.0],
                  "simplex_ok": [True, False, None]}).to_csv(tmp_path / "online_log.csv", index=False)
    app = render(tmp_path)
    assert app.dataframe[0].value["全部通过"].tolist() == ["通过", "未通过", "未知"]
    assert any("未知" in warning.value for warning in app.warning)


def test_online_console_refreshes_after_log_creation_and_append(tmp_path, monkeypatch):
    settings = []
    fragment = st.fragment

    def record_fragment(*args, **kwargs):
        settings.append(kwargs.get("run_every"))
        return fragment(*args, **kwargs)

    monkeypatch.setattr(st, "fragment", record_fragment)
    app = render(tmp_path)
    assert settings[-1] == "2s"
    assert not app.metric
    assert app.info
    app.toggle(key="online_auto_refresh").set_value(False).run()
    assert settings[-1] is None

    path = tmp_path / "online_log.csv"
    path.write_text("call,R\n1,120\n", encoding="utf-8")
    app.button(key="online_refresh").click().run()
    assert labels(app)["已用调用额度"] == "1 / 20"
    assert any("未知" in info.value for info in app.info)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("2,118\n")
    app.button(key="online_refresh").click().run()
    assert not app.exception
    assert labels(app)["已用调用额度"] == "2 / 20"
    assert labels(app)["历史最低 R"] == "118.0000"
    assert chart_values(app)["best_so_far"].tolist() == [120.0, 118.0]


def test_bad_snapshot_hides_stale_metrics_and_recovers_on_refresh(tmp_path):
    path = tmp_path / "online_log.csv"
    path.write_text("call,R\n1,120\n", encoding="utf-8")
    app = render(tmp_path)
    assert app.metric

    path.write_text("call,R\n1,120\n3,118\n", encoding="utf-8")
    app.button(key="online_refresh").click().run()
    assert not app.exception
    assert not app.metric
    assert not app.dataframe
    assert any("连续" in error.value for error in app.error)

    path.write_text("call,R\n1,120\n2,11", encoding="utf-8")
    app.button(key="online_refresh").click().run()
    assert not app.metric
    assert app.error
    with path.open("a", encoding="utf-8") as stream:
        stream.write("8\n")
    app.button(key="online_refresh").click().run()
    assert not app.exception
    assert not app.error
    assert labels(app)["历史最低 R"] == "118.0000"


@pytest.mark.parametrize("calls", [20, 21])
def test_exhausted_quota_is_visible_and_never_becomes_negative(tmp_path, calls):
    pd.DataFrame({"call": range(1, calls + 1), "R": [120.0] * calls}).to_csv(
        tmp_path / "online_log.csv", index=False)
    app = render(tmp_path)
    assert labels(app)["已用调用额度"] == f"{calls} / 20"
    assert labels(app)["剩余调用额度"] == "0"
    if calls == 20:
        assert any("已用完" in warning.value for warning in app.warning)
    else:
        assert any("超过" in error.value for error in app.error)


def test_online_console_follows_the_selected_runs_directory(tmp_path):
    (tmp_path / "online_log.csv").write_text("call,R\n1,120\n", encoding="utf-8")
    app = render(tmp_path)
    assert labels(app)["已用调用额度"] == "1 / 20"
    app.text_input(key="runs_dir").set_value(str(tmp_path / "another-run")).run()
    assert not app.exception
    assert not app.metric
    assert app.info
