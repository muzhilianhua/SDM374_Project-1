import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common
import jobs
from sdm.runs import Run
from sdm.simplex import N_BUILDINGS, N_METRICS, check_weights, project

st.title("实验与数据")

head = st.container(horizontal=True, vertical_alignment="bottom")
head.subheader("训练数据")
if head.button("重新校验", icon=":material/refresh:", help="重新计算 sha256 并检查权重约束"):
    common._load_dataset.clear()
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

# "查看结果" on the launch page hands over the run to open here.
selected = st.session_state.pop("selected_run", None)
if selected in set(shown.run):
    st.session_state.run_choice = selected
choice = st.selectbox("查看运行详情", list(shown.run), key="run_choice")
record = shown.set_index("run").loc[choice]
run = Run(Path(record.path))
meta = run.read_json("meta.json")

if meta["git"].get("dirty"):
    st.warning("记录这次运行时，src/、experiments/ 或锁文件中有未提交的改动，仅凭提交号可能无法复现。",
               icon=":material/warning:")
if not record.complete:
    st.error("运行未完成：缺少 metrics.json（命令出错或被中断）。")
st.markdown("**复现**")
st.code(common.cli_command("reproduce", choice), language="bash")
actions = st.container(horizontal=True)
if actions.button("在后台复现", icon=":material/replay:", key="reproduce_run"):
    job = jobs.start(["reproduce", choice], common.runs_root(), display=common.cli_command("reproduce", choice))
    st.toast(f"已在后台启动：{job.id}", icon=":material/rocket_launch:")
actions.page_link("views/launch.py", label="查看任务进度", icon=":material/list_alt:")

can_predict = run.has("model.pkl")
names = ["指标", "输出", *(["预测"] if can_predict else []), "配置", "来源信息"]
tabs = dict(zip(names, st.tabs(names)))
with tabs["指标"]:
    st.json(record.metrics if record.complete else {}, expanded=True)
with tabs["输出"]:
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
with tabs["配置"]:
    st.code(run.config_path.read_text(encoding="utf-8"), language="toml")
with tabs["来源信息"]:
    st.json(meta, expanded=True)

if can_predict:
    with tabs["预测"]:
        st.caption("编辑 3×7 权重后点击“预测”，用这次运行保存的模型给出预测波动量。每栋楼的 7 项需非负且和为 1。")
        base_key, version_key = f"predict_base_{choice}", f"predict_version_{choice}"
        if base_key not in st.session_state:
            proposals = run.read_json("proposals.json") if run.has("proposals.json") else []
            st.session_state[base_key] = (np.array(proposals[0]["weights"]) if proposals
                                          else np.full((N_BUILDINGS, N_METRICS), 1 / N_METRICS))
            st.session_state[version_key] = 0
        base = common.weights_frame(st.session_state[base_key])
        edited = st.data_editor(base, key=f"predict_editor_{choice}_{st.session_state[version_key]}",
                                column_config={c: st.column_config.NumberColumn(format="%.4f", min_value=0.0)
                                               for c in base.columns}, alt="待预测的 3×7 权重")
        weights = edited.to_numpy(dtype=float)
        sums = weights.sum(axis=1)
        st.caption("各栋行和：" + " · ".join(f"第 {i + 1} 栋 {s:.4f}" for i, s in enumerate(sums)))
        buttons = st.container(horizontal=True)
        if buttons.button("投影到约束", icon=":material/straighten:", key="predict_project"):
            st.session_state[base_key] = project(np.nan_to_num(weights))
            st.session_state[version_key] += 1
            st.rerun()
        if buttons.button("预测", type="primary", icon=":material/insights:", key="predict_run"):
            try:
                check_weights(weights)
            except ValueError as error:
                st.error(f"权重不满足约束：{error}。可以先点击“投影到约束”。")
            else:
                result = jobs.run_now(["predict", "--model-run", choice, "--weights", "-"], common.runs_root(),
                                      input_text=json.dumps(weights.tolist()))
                if result.returncode != 0:
                    st.error(f"预测失败：{result.stderr.strip() or result.stdout.strip()}")
                else:
                    output = json.loads(result.stdout)
                    tiles = st.columns(2)
                    tiles[0].metric("预测波动量", f"{output['predicted_mean'][0]:.2f}", border=True)
                    tiles[1].metric("预测标准差", f"{output['predicted_std'][0]:.2f}", border=True)
