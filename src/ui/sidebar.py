"""Sidebar components."""

import streamlit as st
from ..utils import MODEL_ID, REGION


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.header("⚙️ Cấu hình")

    with st.sidebar.expander("🐛 Debug Mode"):
        show_debug_info = st.checkbox("Show debug info", value=False)
        if show_debug_info:
            st.write("**Config:**")
            st.write(f"- MODEL_ID: `{MODEL_ID}`")
            st.write(f"- REGION: `{REGION}`")
            st.write("Session state:", st.session_state)

    return {"show_debug_info": show_debug_info}