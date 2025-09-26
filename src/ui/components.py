"""UI input and display components."""

import streamlit as st
from PIL import Image
from typing import Optional, Tuple

from ..models import TEXT_MODELS, IMAGE_MODELS, get_default_max_tokens
from ..utils import TEMPERATURE, MAX_TOKENS


def render_input_mode_selector() -> str:
    """Render input mode selection."""
    return st.radio("Chọn chế độ nhập:", ["Text", "Image"], horizontal=True)


def render_model_selector(input_mode: str) -> Tuple[str, str]:
    """Render model selection based on input mode."""
    if input_mode == "Text":
        selected_model_id = st.selectbox(
            "Chọn model cho văn bản:",
            options=list(TEXT_MODELS.keys()),
            format_func=lambda x: TEXT_MODELS[x],
            index=0
        )
        model_name = TEXT_MODELS[selected_model_id]
    else:
        selected_model_id = st.selectbox(
            "Chọn model cho hình ảnh:",
            options=list(IMAGE_MODELS.keys()),
            format_func=lambda x: IMAGE_MODELS[x],
            index=0
        )
        model_name = IMAGE_MODELS[selected_model_id]

    return selected_model_id, model_name


def render_controls(selected_model_id: str) -> Tuple[float, int, bool]:
    """Render control inputs for temperature, max_tokens, and extract button."""
    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input("Temperature", 0.0, 1.0, TEMPERATURE, 0.1)

    with col2:
        default_max_tokens = get_default_max_tokens(selected_model_id, MAX_TOKENS)
        max_tokens = st.number_input("Max tokens", 64, 4096, default_max_tokens, 64)

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_extract = st.button("Extract Ingredients", type="primary")

    return temperature, max_tokens, run_extract


def render_text_input() -> str:
    """Render text input area."""
    return st.text_area(
        "Mô tả món ăn",
        value="Hãy cho tôi nguyên liệu của món phở bò.",
        height=100
    )


def render_image_input() -> Optional[Image.Image]:
    """Render image upload and display."""
    image_file = st.file_uploader(
        "Tải ảnh món ăn (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    img = None
    if image_file is not None:
        img = Image.open(image_file)
        with st.expander("📷 Ảnh đầu vào", expanded=False):
            thumb = img.copy()
            thumb.thumbnail((320, 320))
            st.image(thumb, use_container_width=False)

    return img


def render_validation_warnings(input_mode: str, user_desc: str, img: Optional[Image.Image]) -> bool:
    """Render validation warnings and return True if input is valid."""
    if input_mode == "Text" and not user_desc.strip():
        st.warning("Vui lòng nhập mô tả món ăn.")
        return False
    elif input_mode == "Image" and img is None:
        st.warning("Vui lòng tải ảnh món ăn.")
        return False
    return True