import numpy as np
import streamlit as st

import common
from sdm import catalog

st.title("总览")

records = common.records()
data, data_error = common.dataset()
board = catalog.leaderboard(records)
proposals = catalog.candidates(records)

tiles = st.columns(4)
if data is not None:
    best = int(np.argmin(data.y))
    tiles[0].metric("实测最低波动量", f"{data.y[best]:.2f}", border=True,
                    help=f"训练表中的 {data.ids[best]}，越低越好")
else:
    tiles[0].metric("实测最低波动量", "—", border=True)
if proposals.empty:
    tiles[1].metric("预测最优 W*", "—", border=True)
else:
    top = proposals.iloc[0]
    tiles[1].metric("预测最优 W*", f"{top.predicted_mean:.2f}", border=True,
                    help=f"{top.model} + {top.optimizer} · {top.run}。这是模型预测值，不是实测成绩。")
if board.empty:
    tiles[2].metric("最佳交叉验证 MAE", "—", border=True)
else:
    tiles[2].metric("最佳交叉验证 MAE", f"{board.iloc[0].mae:.3f}", border=True,
                    help=f"{board.iloc[0].model} · {board.iloc[0].run}")
incomplete = int((~records.complete).sum()) if not records.empty else 0
tiles[3].metric("运行次数", len(records), border=True,
                help=f"其中 {incomplete} 次未完成" if incomplete else "全部已完成")

if data_error:
    st.error(f"数据集校验失败：{data_error}")
if records.empty:
    st.info(f"`{common.runs_root()}` 中还没有运行记录。可以先执行 "
            f"`{common.cli_command('evaluate', '-c', 'experiments/baseline.toml')}`。")
    st.stop()

st.subheader("模型排行榜")
st.caption("来自 evaluate 运行：在训练表上做外层 K 折交叉验证，按交叉验证 MAE 从低到高排序。")
if board.empty:
    st.info(f"还没有 evaluate 运行：`{common.cli_command('evaluate', '-c', 'experiments/baseline.toml')}`")
else:
    table = board[["model", "name"] + common.metric_columns(board, ["mae", "rmse", "r2", "spearman"])
                  + ["run", "commit", "dirty"]]
    st.dataframe(table, hide_index=True, column_config=common.column_config(table), alt="模型排行榜")

st.subheader("候选权重")
st.caption("来自 optimize 运行，按预测波动量从低到高排序。距离比 = 到最近训练点的距离 ÷ 训练表中相邻点的典型间距。")
if proposals.empty:
    st.info(f"还没有 optimize 运行：`{common.cli_command('optimize', '-c', 'experiments/baseline.toml')}`")
else:
    flagged = int(proposals.extrapolating.sum())
    if flagged:
        st.warning(f"{len(proposals)} 组候选中有 {flagged} 组离所有训练点都超过典型间距的 "
                   f"{catalog.EXTRAPOLATION_RATIO} 倍。它们的预测波动量属于外推，可能与真实成绩相差很大。",
                   icon=":material/warning:")
    table = proposals[["model", "optimizer", "predicted_mean", "predicted_std", "nearest_train_distance",
                       "distance_ratio", "extrapolating", "run"]]
    st.dataframe(table, hide_index=True, column_config=common.column_config(table), alt="候选权重")

st.subheader("最近的运行")
table = records.iloc[::-1].head(10)[["run", "kind", "model", "optimizer", "created", "commit", "dirty", "complete"]]
st.dataframe(table, hide_index=True, column_config=common.column_config(table), alt="最近 10 次运行")
