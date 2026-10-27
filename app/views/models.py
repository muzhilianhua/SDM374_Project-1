from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import common
from sdm import catalog, registry

st.title("模型与寻优器")
st.caption("列出 `sdm.models` 和 `sdm.optimizers` 中注册的全部插件。新增插件文件后刷新页面即可看到。")

records = common.records()

for title, plugins, field in (("代理模型", registry.models(), "model"),
                              ("寻优器", registry.optimizers(), "optimizer")):
    st.subheader(title)
    for name, cls in sorted(plugins.items()):
        used = int((records[field] == name).sum()) if not records.empty else 0
        with st.container(border=True):
            st.markdown(f"**`{name}`** · `{cls.__module__}.{cls.__qualname__}` · 已用于 {used} 次运行")
            if cls.__doc__:
                st.caption(cls.__doc__.strip())
            # Defaults mix tuples, ints and floats; repr before building the frame, or pandas upcasts 16 to 16.0.
            schema = pd.DataFrame([{**field, "default": repr(field["default"])} for field in registry.describe(cls)],
                                  columns=["name", "type", "default"])
            st.dataframe(schema, hide_index=True, alt=f"{name} 的参数",
                         column_config={"name": "参数", "type": "类型", "default": "默认值"})

st.subheader("比较 evaluate 运行")
board = catalog.leaderboard(records)
if not board.empty:
    board = board[[(Path(path) / "predictions.csv").exists() for path in board.path]]
if board.empty:
    st.info(f"还没有带交叉验证预测的 evaluate 运行：`{common.cli_command('evaluate', '-c', 'experiments/baseline.toml')}`")
    st.stop()

labels = {row.run: f"{row.model} · {row.name} · {row.run}" for row in board.itertuples()}
chosen = st.multiselect("选择运行", list(labels), default=list(labels)[:2], format_func=labels.get)
if not chosen:
    st.stop()

predictions = pd.concat([
    pd.read_csv(Path(board.set_index("run").at[run, "path"]) / "predictions.csv").assign(run=labels[run])
    for run in chosen
])
low = min(predictions.actual.min(), predictions.predicted_mean.min())
high = max(predictions.actual.max(), predictions.predicted_mean.max())
points = alt.Chart(predictions).mark_circle(size=14, opacity=0.35).encode(
    x=alt.X("actual:Q", title="实测波动量", scale=alt.Scale(domain=[low, high])),
    y=alt.Y("predicted_mean:Q", title="交叉验证预测值", scale=alt.Scale(domain=[low, high])),
    color=alt.Color("run:N", title=None, legend=alt.Legend(orient="bottom", direction="vertical")),
    tooltip=[alt.Tooltip("candidate_id", title="候选编号"), alt.Tooltip("actual:Q", title="实测", format=".2f"),
             alt.Tooltip("predicted_mean:Q", title="预测", format=".2f"), alt.Tooltip("run", title="运行")],
)
diagonal = alt.Chart(pd.DataFrame({"actual": [low, high], "predicted_mean": [low, high]})).mark_line(
    strokeDash=[4, 4], color="gray").encode(x="actual:Q", y="predicted_mean:Q")
st.altair_chart(points + diagonal, width="stretch", alt="交叉验证预测值与实测波动量的散点图")
st.caption("落在虚线上的点预测完全准确。对寻优最重要的是左下角：实测波动量低的点，预测值也应该低。")

selected = board[board.run.isin(chosen)]
table = selected[["model", "name", "protocol"]
                 + common.metric_columns(selected, ["v32_score", "v32_local_mae", "v32_top1_regret", "mae", "rmse",
                                                    "r2", "spearman", "mean_predicted_std"])
                 + ["run"]]
st.dataframe(table, hide_index=True, column_config=common.column_config(table), alt="所选运行的评估指标")
