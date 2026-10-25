from datetime import datetime

import streamlit as st

import common
import jobs
from sdm import registry
from sdm.config import load_config, write_new_config
from sdm.paths import REPO_ROOT

COMMANDS = {"evaluate": "评估 evaluate", "train": "训练 train", "optimize": "寻优 optimize"}

st.title("运行实验")
st.caption("每个按钮都会在后台执行一条 `sdm` 命令，和在终端里运行完全相同。结果记录在运行目录中。")

# ---- New config -------------------------------------------------------------------------------
with st.expander("新建实验配置", icon=":material/note_add:"):
    models, optimizers = registry.models(), registry.optimizers()
    left, right = st.columns(2)
    model_name = left.selectbox("代理模型", sorted(models), key="new_model")
    optimizer_name = right.selectbox("寻优器", ["（不设置）", *sorted(optimizers)], key="new_optimizer")
    optimizer_cls = optimizers.get(optimizer_name)
    with st.form("new_config"):
        name = st.text_input("配置名称", placeholder="例如 ridge2-alpha-small",
                             help="只能用小写字母、数字和 -，保存为 experiments/<名称>.toml")
        fields = {"model": registry.describe(models[model_name]),
                  "optimizer": registry.describe(optimizer_cls) if optimizer_cls else []}
        values = {"model": {}, "optimizer": {}}
        for section, title in (("model", f"`{model_name}` 的参数"), ("optimizer", f"`{optimizer_name}` 的参数")):
            if not fields[section]:
                continue
            st.markdown(f"**{title}**")
            columns = st.columns(3)
            for index, field in enumerate(fields[section]):
                values[section][field["name"]] = columns[index % 3].text_input(
                    f"{field['name']}（{field['type']}）", value=common.format_param(field["default"]),
                    key=f"param_{section}_{field['name']}")
        st.caption("评估口径固定为全组统一的 5 折交叉验证（seed 2026，Top-50），这里不能修改。")
        submitted = st.form_submit_button("保存配置", icon=":material/save:")
    if submitted:
        try:
            raw = {"data": {"dataset": "dataset_same8"},
                   "model": {"name": model_name, "params": {
                       f["name"]: common.parse_param(values["model"][f["name"]], f["type"])
                       for f in fields["model"]}},
                   "evaluate": {"folds": 5, "seed": 2026, "top_k": 50}}
            if optimizer_cls:
                raw["optimizer"] = {"name": optimizer_name, "params": {
                    f["name"]: common.parse_param(values["optimizer"][f["name"]], f["type"])
                    for f in fields["optimizer"]}}
            path = write_new_config(name.strip(), raw)
        except (ValueError, TypeError) as error:
            st.error(f"保存失败：{error}")
        else:
            st.session_state.launch_config = f"experiments/{path.name}"
            st.success(f"已保存 `experiments/{path.name}`，并在下方选中。")

# ---- Launch -----------------------------------------------------------------------------------
st.subheader("运行命令")
configs = common.experiment_configs()
if not configs:
    st.info("`experiments/` 中还没有配置，先在上面新建一个。")
    st.stop()
if st.session_state.get("launch_config") not in configs:
    st.session_state.launch_config = "experiments/baseline.toml" if "experiments/baseline.toml" in configs else configs[0]

config_path = st.selectbox("实验配置", configs, key="launch_config")
try:
    config = load_config(REPO_ROOT / config_path)
    config_error = None
except (ValueError, OSError) as error:
    config, config_error = None, str(error)
with st.expander("查看配置内容"):
    st.code((REPO_ROOT / config_path).read_text(encoding="utf-8"), language="toml")

command = st.segmented_control("命令", list(COMMANDS), format_func=COMMANDS.get, default="evaluate",
                               required=True, key="launch_command")
label = st.text_input("运行标签（可选）", key="launch_label", help="默认使用配置文件名",
                      placeholder="留空则使用配置文件名")
args = [command, "-c", config_path]
if label.strip():
    args += ["--name", label.strip()]
if command == "optimize":
    left, right = st.columns(2)
    n = left.number_input("候选数量 -n", min_value=1, max_value=200, value=1, step=1, key="launch_n")
    records = common.records()
    train_runs = [] if records.empty else list(records[(records.kind == "train") & records.complete].run.iloc[::-1])
    model_run = right.selectbox("复用已训练的模型", [None, *train_runs], key="launch_model_run",
                                format_func=lambda run: "不复用，重新训练" if run is None else run)
    args += ["-n", str(int(n))]
    if model_run:
        args += ["--model-run", model_run]

problem = config_error
if not problem and command == "optimize" and config.optimizer is None:
    problem = "这个配置没有 [optimizer] 一节，不能寻优。"
st.markdown("**等价的终端命令**")
st.code(common.cli_command(*args), language="bash")
if problem:
    st.error(problem)
if st.button("运行", type="primary", icon=":material/play_arrow:", disabled=bool(problem), key="launch_run"):
    job = jobs.start(args, common.runs_root(), display=common.cli_command(*args))
    st.toast(f"已在后台启动：{job.id}", icon=":material/rocket_launch:")


# ---- Jobs -------------------------------------------------------------------------------------
@st.fragment(run_every="2s")
def job_panel():
    st.subheader("任务")
    recent = jobs.list_jobs(common.runs_root())[:10]
    if not recent:
        st.caption("还没有任务。")
        return
    for job in recent:
        spec, status = job.spec, job.status
        text, color, icon = common.JOB_STATUS[status]
        with st.container(border=True):
            head = st.container(horizontal=True, vertical_alignment="center")
            head.badge(text, color=color, icon=icon)
            started = datetime.fromisoformat(spec["created"])
            head.caption(f"{job.id} · 开始于 {started:%m-%d %H:%M:%S}")
            st.code(spec["display"], language="bash")
            with st.expander("日志", expanded=status == "running"):
                st.code(job.tail(40) or "（暂无输出）", language="text")
            actions = st.container(horizontal=True)
            if status == "running" and actions.button("取消", key=f"cancel_{job.id}", icon=":material/stop:"):
                jobs.cancel(job)
                st.rerun(scope="fragment")
            run_path = job.run_path
            if status == "succeeded" and run_path and actions.button("查看结果", key=f"open_{job.id}",
                                                                   icon=":material/open_in_new:"):
                st.session_state.selected_run = run_path.name
                st.switch_page("views/experiments.py")


job_panel()
