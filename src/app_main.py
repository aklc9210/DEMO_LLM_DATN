"""Main application orchestrator."""

import streamlit as st
from typing import Optional

from .ui.sidebar import render_sidebar
from .ui.components import (
    render_input_mode_selector, render_model_selector, render_controls,
    render_text_input, render_image_input, render_validation_warnings
)
from .ui.results import render_result
from .inference import invoke_model


class StreamlitApp:
    """Main Streamlit application class."""

    def __init__(self):
        self.bedrock_client = None
        self._initialize_bedrock_client()

    def _initialize_bedrock_client(self):
        """Initialize Bedrock client."""
        try:
            from .bedrock_client import BedrockClient
            self.bedrock_client = BedrockClient()
        except Exception as e:
            st.sidebar.error(f"Không khởi tạo được Bedrock client: {e}")

    def run(self):
        """Run the main application."""
        # Page config
        st.set_page_config(
            page_title="Demo LLM → JSON món ăn",
            page_icon="🍜",
            layout="wide"
        )
        st.title("🍜 Demo: Mô tả món ăn → JSON nguyên liệu")

        # Render sidebar
        sidebar_state = render_sidebar()

        # Main content
        st.subheader("Trích xuất nguyên liệu từ mô tả hoặc hình ảnh")

        # Input mode selection
        input_mode = render_input_mode_selector()

        # Model selection
        selected_model_id, model_name = render_model_selector(input_mode)

        # Controls
        temperature, max_tokens, run_extract = render_controls(selected_model_id)

        # Input based on mode
        user_desc = ""
        img = None

        if input_mode == "Text":
            user_desc = render_text_input()
        else:
            img = render_image_input()

        # Process extraction
        if run_extract:
            self._process_extraction(
                input_mode, user_desc, img, selected_model_id,
                model_name, temperature, max_tokens
            )

        # Footer
        st.divider()
        st.caption("🤖 Powered by AWS Bedrock • Hỗ trợ đa model AI")

    def _process_extraction(
        self, input_mode: str, user_desc: str, img,
        selected_model_id: str, model_name: str,
        temperature: float, max_tokens: int
    ):
        """Process the extraction request."""
        try:
            # Validate inputs
            if not render_validation_warnings(input_mode, user_desc, img):
                return

            # Process with spinner
            with st.spinner(f"Đang xử lý với {model_name}..."):
                if input_mode == "Text":
                    raw_response, metrics = invoke_model(
                        self.bedrock_client, user_desc, selected_model_id,
                        float(temperature), int(max_tokens), prompt_version=3                   )
                else:
                    raw_response, metrics = invoke_model(
                        self.bedrock_client, "", selected_model_id,
                        float(temperature), int(max_tokens), img
                    )

                render_result(raw_response, metrics, model_name)

        except Exception as e:
            st.error(f"❌ Lỗi xử lý: {e}")


def run_app():
    """Entry point for the application."""
    app = StreamlitApp()
    app.run()