"""SDM374 workbench: `uv run --group gui streamlit run app/main.py`.

Views runs/ and the plugin registry, and launches `sdm` CLI commands as background jobs (app/jobs.py).
"""
import streamlit as st

import common
import jobs
from sdm.paths import DEFAULT_RUNS_DIR

st.set_page_config(page_title="SDM374 工作台", page_icon=":material/bolt:", layout="wide")

if "runs_dir" not in st.session_state:
    st.session_state.runs_dir = str(DEFAULT_RUNS_DIR)

page = st.navigation([
    st.Page("views/overview.py", title="总览", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page("views/launch.py", title="运行实验", icon=":material/play_circle:", url_path="launch"),
    st.Page("views/models.py", title="模型", icon=":material/model_training:", url_path="models"),
    st.Page("views/experiments.py", title="实验与数据", icon=":material/science:", url_path="experiments"),
])

with st.sidebar:
    st.text_input("运行目录", key="runs_dir")
    running = sum(job.status == jobs.RUNNING for job in jobs.list_jobs(common.runs_root()))
    if running:
        st.caption(f":material/progress_activity: {running} 个任务正在运行，详见“运行实验”页。")
    st.caption("训练、评估、寻优、复现和预测都会调用 `sdm` 命令行，结果与终端运行完全一致。")

page.run()
