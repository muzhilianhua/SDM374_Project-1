"""Read-only dashboard for the append-only online evaluator log."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import common
from online_log import check_columns, read_online_log

MAX_CALLS = 20


st.title("在线控制台")
st.caption("只读显示在线评估器的额度、提交前检查和每批最低 R；所有计算仍由在线 CLI 完成。")

log_path = common.runs_root() / "online_log.csv"
log, error = read_online_log(log_path)
if error:
    st.error(error)
    st.stop()
if log is None:
    st.info(f"尚未找到 `{log_path}`。在线主程序开始写入 T07 日志后，这里会自动显示进度。")
    st.stop()

calls = sorted(log["_call"].unique())
used = len(calls)
tiles = st.columns(3)
tiles[0].metric("已用调用额度", f"{used} / {MAX_CALLS}", border=True)
tiles[1].metric("已记录权重组", len(log), border=True)
tiles[2].metric("历史最低 R", f"{log['_r'].min():.4f}", border=True)
if used > MAX_CALLS:
    st.error(f"日志记录了 {used} 次调用，已超过正式额度 {MAX_CALLS}。")

checks = check_columns(log)
st.subheader("提交前检查")
if not checks:
    st.info("当前日志没有检查字段；在线主程序写入 `check_*` 或 `*_ok` 列后会在这里汇总。")
else:
    summary = []
    for call in calls:
        batch = log.loc[log["_call"] == call]
        values = {}
        for column in checks:
            parsed = batch[column].map(lambda value: str(value).strip().casefold() in {"1", "true", "yes", "ok"})
            values[column] = bool(parsed.all())
        summary.append({"调用号": call, **values, "全部通过": all(values.values())})
    checks_frame = pd.DataFrame(summary)
    st.dataframe(checks_frame, hide_index=True, column_config=common.column_config(checks_frame),
                 alt="每次调用的提交前检查")

st.subheader("逐次最优值")
curve = log.groupby("_call", as_index=False)["_r"].min().rename(columns={"_call": "call", "_r": "batch_best"})
curve["best_so_far"] = curve["batch_best"].cummin()
chart = alt.Chart(curve).mark_line(point=True).encode(
    x=alt.X("call:Q", title="调用号"),
    y=alt.Y("best_so_far:Q", title="截至当前的最低 R"),
    tooltip=[alt.Tooltip("call:Q", title="调用"),
             alt.Tooltip("batch_best:Q", title="本批最低 R", format=".4f"),
             alt.Tooltip("best_so_far:Q", title="历史最低 R", format=".4f")],
)
st.altair_chart(chart, width="stretch", alt="逐次历史最低 R 曲线")

st.subheader("最近一批")
latest = log[log["_call"] == calls[-1]].drop(columns=["_call", "_r"])
st.dataframe(latest, hide_index=True, column_config=common.column_config(latest), alt="最近一批在线日志")
