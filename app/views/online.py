"""T11: read-only online quota, pre-submit evidence and best-so-far curve."""
from __future__ import annotations

from datetime import datetime
import inspect

import altair as alt
import streamlit as st

import common
from online_log import MAX_CALLS
from online_run import discover_online_runs, read_run


st.title("在线控制台")
st.caption("查看在线运行目录的调用额度、提交前检查结果和最优值曲线。评估由在线主程序执行。")
controls = st.columns(2)
auto_refresh = controls[0].toggle("自动刷新", value=True, key="online_auto_refresh")
controls[1].button("立即刷新", key="online_refresh", icon=":material/refresh:")
st.caption("每 2 秒读取最新运行目录。" if auto_refresh else "自动刷新已暂停，可点击“立即刷新”更新。")


@st.fragment(run_every="2s" if auto_refresh else None)
def online_panel():
    root = common.runs_root()
    runs = discover_online_runs(root)
    if not runs:
        st.info(f"运行目录 `{root}` 中暂无在线运行。在线主程序创建 run 文件夹后，这里会显示进度。")
        return
    if len(runs) == 1:
        run_path = runs[0]
    else:
        labels = [path.name for path in runs]
        choice = st.selectbox("在线运行", labels, index=0, key="online_run_selection")
        run_path = runs[labels.index(choice)]
    try:
        updated = datetime.fromtimestamp(run_path.stat().st_mtime).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z")
    except OSError:
        updated = "不可用"
    st.caption(f"运行目录：`{run_path}` · 最后更新：{updated}")
    snapshot = read_run(run_path)
    if snapshot.error:
        st.error(snapshot.error)
        st.info("本次刷新不显示旧额度和曲线，待日志与台账形成完整快照后会自动恢复。")
        return
    if snapshot.log is None and snapshot.used is None:
        st.info("暂无在线记录。在线主程序写入日志后，这里会显示进度。")
        return

    used = snapshot.used
    budget = snapshot.budget or MAX_CALLS
    tiles = st.columns(4)
    tiles[0].metric("已用调用额度", f"{used} / {budget}", border=True)
    tiles[1].metric("剩余调用额度", max(0, budget - used), border=True)
    tiles[2].metric("已记录权重组", len(snapshot.log) if snapshot.log is not None else 0, border=True)
    tiles[3].metric("历史最低 R", "—" if snapshot.log is None else f"{snapshot.log['_r'].min():.4f}", border=True)
    if used > budget:
        st.error(f"运行台账记录了 {used} 次调用，已超过正式额度 {budget}。")
    elif used == budget:
        st.warning(f"{budget} 次调用额度已用完。")
    if snapshot.journal and snapshot.journal["pending"]:
        st.warning(f"有未闭合的调用尝试：{snapshot.journal['pending']}；额度已按最安全口径计入。")
    if snapshot.journal is None:
        st.warning("旧版日志仅包含成功调用；失败／丢失尝试未知，以上额度仅按已记录调用估算。")
    else:
        st.caption(f"额度来源：attempts.jsonl；失败／丢失 {snapshot.journal['failed']} 次，"
                   f"待闭合 {len(snapshot.journal['pending'])} 次。")

    st.subheader("提交前检查")
    checks = snapshot.checks
    if checks.empty:
        st.info("运行记录尚未提供逐次提交前检查，当前检查状态未知。")
    else:
        if "全部通过" in checks and (checks["全部通过"] != "通过").any():
            st.warning("部分调用存在未通过或未知的检查结果，请核对下表。")
        st.dataframe(checks, hide_index=True, column_config=common.column_config(checks))

    if snapshot.log is None:
        return
    st.subheader("逐次最优值")
    curve = (snapshot.log.groupby("_call", as_index=False)["_r"].min()
             .rename(columns={"_call": "call", "_r": "batch_best"}))
    curve["best_so_far"] = curve["batch_best"].cummin()
    chart = alt.Chart(curve).mark_line(point=True).encode(
        x=alt.X("call:Q", title="调用号"),
        y=alt.Y("best_so_far:Q", title="截至当前的最低 R", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("call:Q", title="调用"),
                 alt.Tooltip("batch_best:Q", title="本批最低 R", format=".4f"),
                 alt.Tooltip("best_so_far:Q", title="历史最低 R", format=".4f")],
    )
    if "alt" in inspect.signature(st.altair_chart).parameters:
        st.altair_chart(chart, width="stretch", alt="逐次历史最低 R 曲线")
    else:
        st.altair_chart(chart, use_container_width=True)

    st.subheader("最近一批")
    latest = snapshot.log[snapshot.log["_call"] == int(snapshot.log["_call"].max())].drop(
        columns=["_call", "_r"])
    st.dataframe(latest, hide_index=True, column_config=common.column_config(latest))


online_panel()
