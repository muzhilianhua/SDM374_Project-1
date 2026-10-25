from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import common
from sdm.runs import Run

st.title("实验与数据")

st.subheader("训练数据")
data, error = common.dataset()
if data is None:
    st.error(f"数据集校验失败：{error}")
else:
    st.success(f"`{data.version}` · {len(data.y)} 行 · sha256 与 manifest 一致 · 权重约束全部满足",
               icon=":material/verified:")
    st.caption(f"sha256 `{data.sha256}`")
    table = pd.DataFrame({"candidate_id": data.ids, "ramping": data.y})
    left, right = st.columns([3, 2])
    with left:
        histogram = alt.Chart(table).mark_bar().encode(
            x=alt.X("ramping:Q", bin=alt.Bin(maxbins=40), title="第 1–8 周平均波动量"),
            y=alt.Y("count():Q", title="行数"),
        ).properties(height=260)
        st.altair_chart(histogram, width="stretch", alt="训练表波动量分布直方图")
    with right:
        st.markdown("**训练表中波动量最低的 10 行**")
        lowest = table.nsmallest(10, "ramping")
        st.dataframe(lowest, hide_index=True, column_config=common.column_config(lowest), alt="波动量最低的 10 行")

st.subheader("运行记录")
records = common.records()
if records.empty:
    st.info(f"`{common.runs_root()}` 中还没有运行记录。")
    st.stop()

kinds = sorted(records.kind.unique())
kinds = st.multiselect("类型", kinds, default=kinds)
shown = records[records.kind.isin(kinds)].iloc[::-1]
table = shown[["run", "kind", "name", "model", "optimizer", "model_run", "created", "commit", "dirty", "complete"]]
st.dataframe(table, hide_index=True, column_config=common.column_config(table), alt="运行记录")
if shown.empty:
    st.stop()

choice = st.selectbox("查看运行详情", list(shown.run))
record = shown.set_index("run").loc[choice]
run = Run(Path(record.path))
meta = run.read_json("meta.json")

if meta["git"].get("dirty"):
    st.warning("记录这次运行时，src/、experiments/ 或锁文件中有未提交的改动，仅凭提交号可能无法复现。",
               icon=":material/warning:")
if not record.complete:
    st.error("运行未完成：缺少 metrics.json（命令出错或被中断）。")
st.markdown("**复现命令**")
st.code(common.cli_command("reproduce", choice), language="bash")

tabs = st.tabs(["指标", "输出", "配置", "来源信息"])
with tabs[0]:
    st.json(record.metrics if record.complete else {}, expanded=True)
with tabs[1]:
    if run.has("weights.json"):
        weights = run.read_json("weights.json")
        if meta["options"].get("n", 1) == 1:
            weights = [weights]
        proposals = run.read_json("proposals.json") if run.has("proposals.json") else []
        for index, matrix in enumerate(weights):
            title = f"候选 {index + 1}"
            if index < len(proposals):
                title += f" · 预测波动量 {proposals[index]['predicted_mean']:.2f}"
            st.markdown(f"**{title}**")
            frame = common.weights_frame(matrix)
            st.dataframe(frame, column_config=common.column_config(frame), alt=f"候选 {index + 1} 的 3×7 权重")
    elif run.has("predictions.csv"):
        predictions = pd.read_csv(run.path / "predictions.csv")
        st.dataframe(predictions, hide_index=True, column_config=common.column_config(predictions),
                     alt="交叉验证预测结果")
    elif run.has("model.pkl"):
        st.markdown(f"拟合好的模型保存为 `model.pkl`，可以这样复用："
                    f"`{common.cli_command('optimize', '-c', 'experiments/<config>.toml', '--model-run', choice)}`")
    else:
        st.caption("这次运行没有输出文件。")
with tabs[2]:
    st.code(run.config_path.read_text(encoding="utf-8"), language="toml")
with tabs[3]:
    st.json(meta, expanded=True)
