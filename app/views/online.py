"""Read-only dashboard for the append-only online evaluator log."""
from __future__ import annotations

from datetime import datetime

import altair as alt
import streamlit as st

import common
from online_log import MAX_CALLS, read_online_log, summarize_checks


st.title("在线控制台")
st.caption("查看在线日志中的调用额度、检查结果和最优值曲线。评估由在线主程序执行。")
controls = st.columns(2)
auto_refresh = controls[0].toggle("自动刷新", value=True, key="online_auto_refresh")
controls[1].button("立即刷新", key="online_refresh", icon=":material/refresh:")
st.caption("每 2 秒读取最新日志。" if auto_refresh else "自动刷新已暂停，可点击“立即刷新”更新。")


@st.fragment(run_every="2s" if auto_refresh else None)
def online_panel():
    log_path = common.runs_root() / "online_log.csv"
    try:
        updated = datetime.fromtimestamp(log_path.stat().st_mtime).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z")
    except OSError:
        updated = "不可用"
    st.caption(f"日志位置：`{log_path}` · 最后更新：{updated}")
    log, error = read_online_log(log_path)
    if error:
        st.error(error)
        st.info("本次日志无法可靠统计额度。请核对日志；下次刷新会重新读取。")
        return
    if log is None:
        st.info("暂无在线记录。在线主程序写入日志后，这里会显示进度。")
        return

    # The reader requires a contiguous history starting at call 1. Use the last call number
    # explicitly so the displayed quota refers to calls, rather than the number of result rows.
    used = int(log["_call"].max())
    tiles = st.columns(4)
    tiles[0].metric("已用调用额度", f"{used} / {MAX_CALLS}", border=True)
    tiles[1].metric("剩余调用额度", max(0, MAX_CALLS - used), border=True)
    tiles[2].metric("已记录权重组", len(log), border=True)
    tiles[3].metric("历史最低 R", f"{log['_r'].min():.4f}", border=True)
    if used > MAX_CALLS:
        st.error(f"日志记录了 {used} 次调用，已超过正式额度 {MAX_CALLS}。")
    elif used == MAX_CALLS:
        st.warning("20 次调用额度已用完。")

    st.subheader("提交前检查")
    checks = summarize_checks(log)
    if checks.empty:
        st.info("日志尚未记录提交前检查，当前检查状态未知。")
    else:
        if (checks["全部通过"] != "通过").any():
            st.warning("部分调用存在未通过或未知的检查结果，请核对下表。")
        st.dataframe(checks, hide_index=True, column_config=common.column_config(checks),
                     alt="每次调用的提交前检查")

    st.subheader("逐次最优值")
    curve = (log.groupby("_call", as_index=False)["_r"].min()
             .rename(columns={"_call": "call", "_r": "batch_best"}))
    curve["best_so_far"] = curve["batch_best"].cummin()
    chart = alt.Chart(curve).mark_line(point=True).encode(
        x=alt.X("call:Q", title="调用号"),
        y=alt.Y("best_so_far:Q", title="截至当前的最低 R", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("call:Q", title="调用"),
                 alt.Tooltip("batch_best:Q", title="本批最低 R", format=".4f"),
                 alt.Tooltip("best_so_far:Q", title="历史最低 R", format=".4f")],
    )
    st.altair_chart(chart, width="stretch", alt="逐次历史最低 R 曲线")

    st.subheader("最近一批")
    latest = log[log["_call"] == used].drop(columns=["_call", "_r"])
    st.dataframe(latest, hide_index=True, column_config=common.column_config(latest),
                 alt="最近一批在线日志")


online_panel()
