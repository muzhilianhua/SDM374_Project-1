"""SDM374 workbench: `uv run --group gui streamlit run app/main.py`.

Read-only view of runs/ and the plugin registry; runs are created with the `sdm` CLI.
"""
import streamlit as st

from sdm.paths import DEFAULT_RUNS_DIR

st.set_page_config(page_title="SDM374 工作台", page_icon=":material/bolt:", layout="wide")

if "runs_dir" not in st.session_state:
    st.session_state.runs_dir = str(DEFAULT_RUNS_DIR)

page = st.navigation([
    st.Page("views/overview.py", title="总览", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page("views/models.py", title="模型", icon=":material/model_training:", url_path="models"),
    st.Page("views/experiments.py", title="实验与数据", icon=":material/science:", url_path="experiments"),
])

with st.sidebar:
    st.text_input("运行目录", key="runs_dir")
    st.caption("只读界面。用 `uv run sdm train | evaluate | optimize` 新建运行后刷新页面。")

page.run()
