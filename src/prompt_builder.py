import json
from .schema import DISH_JSON_SCHEMA

# =========================
# System prompts 
# =========================
SYSTEM_INSTRUCTIONS = (
    "Bạn là trợ lý ẩm thực chuyên trích xuất nguyên liệu.\n"
    "Yêu cầu: trả về DUY NHẤT một JSON hợp lệ theo schema đã cho.\n"
    "Không trả lời giải thích, không thêm text ngoài JSON.\n"
    "Nếu thông tin không chắc chắn, để trống hoặc bỏ qua trường đó thay vì bịa.\n"
    "Ngôn ngữ ưu tiên: giữ tiếng Việt cho tên nguyên liệu.\n"
)

SYSTEM_INSTRUCTIONS_IMAGE = (
    "Bạn là trợ lý ẩm thực chuyên trích xuất nguyên liệu TỪ ẢNH.\n"
    "Chỉ trả về DUY NHẤT một JSON theo schema đã cho (không thêm chữ/thuyết minh/markdown).\n"
    "Quy tắc PHÂN LOẠI & ĐIỀN TRƯỜNG:\n"
    "1) Nếu ảnh KHÔNG chứa món ăn hoặc nguyên liệu (none):\n"
    "   - Trả về JSON với: dish_name = null, cuisine = null, ingredients = []\n"
    "2) Nếu ảnh CHỈ chứa NGUYÊN LIỆU riêng lẻ (ingredient):\n"
    "   - Trả về JSON với: dish_name = null, cuisine = null\n"
    "   - Điền danh sách ingredients nếu nhận diện được (ưu tiên tên cụ thể, ví dụ 'thịt bò thăn', 'rau mùi').\n"
    "3) Nếu ảnh chứa MÓN ĂN (dish):\n"
    "   - BẮT BUỘC điền cả 'dish_name' và 'cuisine' (không được bỏ trống)\n"
    "   - Kèm danh sách 'ingredients' như thường lệ.\n"
    "   - Nếu hai nguyên liệu trùng nhau về tên: chỉ giữ một mục, bỏ bản sao.\n"
    "Yêu cầu về định dạng:\n"
    "- 'quantity' là chuỗi số; 'unit' là chuỗi đơn vị phù hợp hoặc null.\n"
    "- Tránh tên chung chung như 'thịt', 'rau' - hãy cụ thể.\n"
    "- Nếu không có đơn vị rõ ràng, để unit = null (không bịa).\n"
    "Ngôn ngữ ưu tiên: giữ tiếng Việt cho tên nguyên liệu.\n"
)

# Ví dụ tham khảo 
FEW_SHOT_EXAMPLE = {
    "dish_name": "Phở bò",
    "cuisine": "Vietnamese",
    "ingredients": [
        {"name": "bánh phở", "quantity": "200", "unit": "g"},
        {"name": "thịt bò thăn", "quantity": "250", "unit": "g"},
        {"name": "hành lá", "quantity": "2", "unit": "nhánh"},
        {"name": "quế", "quantity": "1", "unit": "thanh"},
        {"name": "gừng", "quantity": "1", "unit": "củ"},
        {"name": "rau mùi", "quantity": "1", "unit": "ít"},
    ],
    "notes": ["Định lượng có thể thay đổi theo khẩu vị"],
}

# =========================
# Prompt mặc định 
# =========================
def build_user_text(user_dish_description: str, include_example: bool = True) -> str:
    """User text mặc định (prompt gốc)."""
    schema_str = json.dumps(DISH_JSON_SCHEMA, ensure_ascii=False)

    user_text = f"""
    Nhiệm vụ: Từ mô tả sau, hãy xuất JSON nguyên liệu theo đúng schema.

    Mô tả món ăn:
    \"\"\"{user_dish_description}\"\"\"

    Yêu cầu định dạng:
    - Trả về duy nhất một JSON.
    - Tuân thủ schema (bên dưới) cả về key và kiểu.
    - QUAN TRỌNG: Tách riêng số lượng và đơn vị:
        + "quantity": chỉ chứa SỐ (ví dụ: "200", "1", "2", "0.5")
        + "unit": chỉ chứa ĐƠN VỊ thích hợp cho nguyên liệu đó
    - Hướng dẫn về đơn vị:
        + Thịt, cá, rau củ quả: "g" hoặc "kg" 
        + Chất lỏng (nước, dầu, sữa): "ml" hoặc "l"
        + Rau lá, gia vị: "nhánh", "lá", "ít", "chút"
        + Củ quả đếm được: "củ", "quả", "trái"
        + Tỏi: "tép", "củ"
        + Hành: "củ", "nhánh"
        + Trứng: "quả", "lòng đỏ", "lòng trắng"
        + Bánh mì, bánh phở: "g" hoặc "gói"
        + Gia vị khô: "g", "muống", "thìa"
    - CHỈ để unit = null khi thực sự không xác định được đơn vị phù hợp

    Schema (JSON Schema):
    {schema_str}"""

    if include_example:
        example_str = json.dumps(FEW_SHOT_EXAMPLE, ensure_ascii=False)
        user_text += f"""

    Ví dụ (chỉ tham khảo, KHÔNG lẫn vào output):
    {example_str}"""

        user_text += """

    Lưu ý: 
    - Quantity LUÔN là chuỗi số ("1", "200", "0.5")
    - Unit nên là đơn vị thích hợp ("g", "ml", "củ", "nhánh", "ít") - tránh để null trừ khi thực sự không biết
    - Ưu tiên suy luận đơn vị phù hợp dựa trên loại nguyên liệu
    - Trước khi trả: tự kiểm tra JSON hợp lệ theo schema. Nếu chưa hợp lệ, tự sửa rồi mới trả."""

    return user_text.strip()

# ---- Claude (Anthropic) - mặc định ----
def build_prompt(user_dish_description: str, temperature: float = 0.2, max_tokens: int = 512):
    user_text = build_user_text(user_dish_description)
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_INSTRUCTIONS,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

# ---- Titan Text - mặc định ----
def build_prompt_titan(user_dish_description: str, temperature: float = 0.2, max_tokens: int = 512):
    user_text = build_user_text(user_dish_description)
    guard = ("CHỈ TRẢ 1 JSON hợp lệ theo schema. "
             "KHÔNG in schema/giải thích/markdown. "
             "'dish_name' bắt buộc có giá trị (không rỗng).")
    example = ('Ví dụ (tham khảo, KHÔNG lẫn vào output): '
               '{"dish_name":"Phở bò","cuisine":"Vietnamese","ingredients":[{"name":"bánh phở","quantity":"200","unit":"g"}]}')
    return {
        "inputText": f"{SYSTEM_INSTRUCTIONS}\n\n{guard}\n\n{example}\n\n{user_text}",
        "textGenerationConfig": {
            "temperature": temperature, "topP": 0.9,
            "maxTokenCount": max_tokens, "stopSequences": []
        },
    }

# ---- Llama - mặc định ----
def build_prompt_llama(user_dish_description: str, temperature: float = 0.2, max_tokens: int = 512):
    user_text = build_user_text(user_dish_description, include_example=False)
    prompt_text = f"{SYSTEM_INSTRUCTIONS}\n\n{user_text}"
    return {"prompt": prompt_text, "temperature": temperature, "top_p": 0.9, "max_gen_len": max_tokens}

# ---- Nova (messages-v1) - mặc định ----
def build_prompt_nova(user_dish_description: str, temperature: float = 0.2, max_tokens: int = 512):
    user_text = build_user_text(user_dish_description)
    return {
        "schemaVersion": "messages-v1",
        "system": [{"text": SYSTEM_INSTRUCTIONS}],
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "inferenceConfig": {"temperature": temperature, "topP": 0.9, "maxTokens": max_tokens},
    }

# =========================
# Prompt dành cho ẢNH (giữ nguyên)
# =========================
def build_prompt_with_image(
    user_dish_description: str,
    image_b64: str,
    image_mime: str = "image/png",
    temperature: float = 0.2,
    max_tokens: int = 512,
):
    schema_str = json.dumps(DISH_JSON_SCHEMA, ensure_ascii=False)
    example_str = json.dumps(FEW_SHOT_EXAMPLE, ensure_ascii=False)

    user_text = f"""Hãy PHÂN LOẠI ảnh thành một trong ba loại: dish | ingredient | none.
                    QUY TẮC:
                    - none: trả về {{ "dish_name": null, "cuisine": null, "ingredients": [] }}
                    - ingredient: trả về {{ "dish_name": null, "cuisine": null, "ingredients": [...] }}
                    dish: BẮT BUỘC có 'dish_name' và 'cuisine', kèm 'ingredients'

                    Mô tả bổ sung (nếu có):
                    \"\"\"{user_dish_description}\"\"\"\n
                    ĐỊNH DẠNG & RÀNG BUỘC:
                    - JSON DUY NHẤT theo schema (không thêm giải thích).
                    - quantity = chuỗi số; unit = chuỗi đơn vị hoặc null.
                    - Ưu tiên tên cụ thể cho nguyên liệu.
                    - Nếu không chắc đơn vị: để unit = null.

                    Schema (JSON Schema):
                    {schema_str}

                    Ví dụ (tham khảo, KHÔNG lẫn vào output):
                    {example_str}

                    Tự kiểm tra JSON hợp lệ theo schema trước khi trả."""
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_INSTRUCTIONS_IMAGE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": image_mime, "data": image_b64}},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

def build_prompt_nova_with_image(
    user_dish_description: str,
    image_b64: str,
    image_mime: str = "image/png",
    temperature: float = 0.2,
    max_tokens: int = 512,
):
    fmt = "png"
    lmime = (image_mime or "").lower()
    if lmime in ("image/jpeg", "image/jpg"):
        fmt = "jpeg"
    elif lmime == "image/webp":
        fmt = "webp"
    elif lmime == "image/gif":
        fmt = "gif"

    schema_str = json.dumps(DISH_JSON_SCHEMA, ensure_ascii=False)
    example_str = json.dumps(FEW_SHOT_EXAMPLE, ensure_ascii=False)

    user_text = f"""PHÂN LOẠI ảnh: dish | ingredient | none.
                QUY TẮC ĐIỀN JSON:
                - none ⇒ dish_name = null, cuisine = null, ingredients = []
                - ingredient ⇒ dish_name = null, cuisine = null, có thể liệt kê ingredients nếu nhận ra
                - dish ⇒ BẮT BUỘC có dish_name và cuisine, kèm ingredients

                RÀNG BUỘC ĐỊNH DẠNG:
                - Trả DUY NHẤT 1 JSON theo schema.
                - quantity là chuỗi số; unit là chuỗi đơn vị hoặc null; không bịa.
                - Ưu tiên tên nguyên liệu cụ thể.

                Schema:
                {schema_str}

                Ví dụ (tham khảo, KHÔNG lẫn vào output):
                {example_str}

                Mô tả bổ sung (nếu có):
                \"\"\"{user_dish_description}\"\"\""""
    return {
        "schemaVersion": "messages-v1",
        "system": [{"text": SYSTEM_INSTRUCTIONS_IMAGE}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": user_text},
                    {"image": {"format": fmt, "source": {"bytes": image_b64}}},
                ],
            }
        ],
        "inferenceConfig": {"temperature": temperature, "topP": 0.9, "maxTokens": max_tokens},
    }

# =========================
# Prompt Version 1 / 2 / 3 (TEXT) cho từng model
# =========================

def _user_text_v1(desc: str) -> str:
    schema_str = json.dumps(DISH_JSON_SCHEMA, ensure_ascii=False)
    return (
        "Trích xuất NGUYÊN LIỆU từ mô tả món ăn thành JSON.\n\n"
        f'Input: """{desc}"""\n\n'
        "Yêu cầu:\n"
        "- Chỉ trả JSON, không giải thích\n"
        '- quantity: chỉ số ("200", "1")\n'
        '- unit: đơn vị phù hợp ("g", "ml", "củ", "nhánh", "ít") - tránh null\n'
        "- Giữ tên tiếng Việt\n\n"
        f"Schema: {schema_str}\n\n"
        'Ví dụ: {"dish_name":"Phở bò","ingredients":[{"name":"bánh phở","quantity":"200","unit":"g"}]}'
    ).strip()

def _user_text_v2(desc: str) -> str:
    schema_str = json.dumps(DISH_JSON_SCHEMA, ensure_ascii=False)
    return (
        "Trích xuất nguyên liệu → JSON. Chỉ trả JSON.\n\n"
        f'Input: """{desc}"""\n\n'
        "Format:\n"
        '- quantity: số dạng string ("200")\n'
        '- unit: đơn vị string ("g") hoặc null\n'
        "- Tên tiếng Việt\n\n"
        f"Schema: {schema_str}"
    ).strip()

def _user_text_v3(desc: str) -> str:
    schema_str = json.dumps(DISH_JSON_SCHEMA, ensure_ascii=False)
    return (
        "TASK: Extract ingredients → JSON\n"
        f'INPUT: """{desc}"""\n'
        "OUTPUT: JSON only, no explanation\n\n"
        "RULES:\n"
        '- quantity: number as string ("200", "1")\n'
        '- unit: unit string ("g", "củ") or null\n'
        "- Vietnamese ingredient names\n"
        f"- Follow schema: {schema_str}\n\n"
        'EXAMPLE: {"ingredients":[{"name":"gạo","quantity":"1","unit":"kg"}]}'
    ).strip()

SYSTEM_SHORT = (
    "Bạn là trợ lý ẩm thực. Trả về DUY NHẤT một JSON hợp lệ theo schema. "
    "Không thêm giải thích. Tên nguyên liệu giữ tiếng Việt. "
    "quantity là chuỗi số; unit là chuỗi đơn vị hoặc null."
)

# ---- Claude (Anthropic) V1/V2/V3 ----
def build_prompt_v1(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_SHORT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": _user_text_v1(desc)}]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

def build_prompt_v2(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_SHORT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": _user_text_v2(desc)}]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

def build_prompt_v3(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_SHORT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": _user_text_v3(desc)}]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

# ---- Titan V1/V2/V3 ----
GUARD = (
  "CHỈ TRẢ 1 JSON dữ liệu hợp lệ theo schema. "
  "KHÔNG in schema/giải thích/markdown. "
  "'dish_name' bắt buộc có giá trị (không rỗng)."
)
EXAMPLE = ('Ví dụ (tham khảo, KHÔNG lẫn vào output): '
           '{"dish_name":"Phở bò","cuisine":"Vietnamese","ingredients":[{"name":"bánh phở","quantity":"200","unit":"g"}]}')


def build_prompt_titan_v1(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    return {
        # "inputText": f"{SYSTEM_SHORT}\n\n{_user_text_v1(desc)}",
        "inputText": f"{SYSTEM_SHORT}\n\n{GUARD}\n\n{EXAMPLE}\n\n{_user_text_v1(desc)}",
        "textGenerationConfig": {"temperature": temperature, "topP": 0.9, "maxTokenCount": max_tokens, "stopSequences": []},
    }

def build_prompt_titan_v2(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    return {
        # "inputText": f"{SYSTEM_SHORT}\n\n{_user_text_v2(desc)}",
        "inputText": f"{SYSTEM_SHORT}\n\n{GUARD}\n\n{EXAMPLE}\n\n{_user_text_v2(desc)}",
        "textGenerationConfig": {"temperature": temperature, "topP": 0.9, "maxTokenCount": max_tokens, "stopSequences": []},
    }

def build_prompt_titan_v3(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    return {
        # "inputText": f"{SYSTEM_SHORT}\n\n{_user_text_v3(desc)}",
        "inputText": f"{SYSTEM_SHORT}\n\n{GUARD}\n\n{EXAMPLE}\n\n{_user_text_v3(desc)}",
        "textGenerationConfig": {"temperature": temperature, "topP": 0.9, "maxTokenCount": max_tokens, "stopSequences": []},
    }

# ---- Llama V1/V2/V3 ----
def build_prompt_llama_v1(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    focus_instruction = (
        "LƯU Ý QUAN TRỌNG:\n"
        "- 'dish_name' LUÔN phải có tên món ăn, không được để trống hoặc rỗng.\n"
        "- Nếu không chắc chắn, hãy dùng mô tả món ăn của người dùng để điền 'dish_name'.\n"
    )
    return {
        "prompt": f"{SYSTEM_SHORT}\n\n{focus_instruction}\n\n{_user_text_v1(desc)}",
        "temperature": temperature,
        "top_p": 0.9,
        "max_gen_len": max_tokens,
    }


def build_prompt_llama_v2(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    focus_instruction = (
        "LƯU Ý QUAN TRỌNG:\n"
        "- 'dish_name' LUÔN phải có tên món ăn, không được để trống hoặc rỗng.\n"
        "- Nếu không chắc chắn, hãy dùng mô tả món ăn của người dùng để điền 'dish_name'.\n"
    )
    return {
        "prompt": f"{SYSTEM_SHORT}\n\n{focus_instruction}\n\n{_user_text_v2(desc)}",
        "temperature": temperature,
        "top_p": 0.9,
        "max_gen_len": max_tokens,
    }


def build_prompt_llama_v3(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    focus_instruction = (
        "LƯU Ý QUAN TRỌNG:\n"
        "- 'dish_name' LUÔN phải có tên món ăn, không được để trống hoặc rỗng.\n"
        "- Nếu không chắc chắn, hãy dùng mô tả món ăn của người dùng để điền 'dish_name'.\n"
    )
    return {
        "prompt": f"{SYSTEM_SHORT}\n\n{focus_instruction}\n\n{_user_text_v3(desc)}",
        "temperature": temperature,
        "top_p": 0.9,
        "max_gen_len": max_tokens,
    }


# ---- Nova V1/V2/V3 (messages-v1) ----
def build_prompt_nova_v1(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    return {
        "schemaVersion": "messages-v1",
        "system": [{"text": SYSTEM_SHORT}],
        "messages": [{"role": "user", "content": [{"text": _user_text_v1(desc)}]}],
        "inferenceConfig": {"temperature": temperature, "topP": 0.9, "maxTokens": max_tokens},
    }

def build_prompt_nova_v2(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    return {
        "schemaVersion": "messages-v1",
        "system": [{"text": SYSTEM_SHORT}],
        "messages": [{"role": "user", "content": [{"text": _user_text_v2(desc)}]}],
        "inferenceConfig": {"temperature": temperature, "topP": 0.9, "maxTokens": max_tokens},
    }

def build_prompt_nova_v3(desc: str, temperature: float = 0.2, max_tokens: int = 512):
    return {
        "schemaVersion": "messages-v1",
        "system": [{"text": SYSTEM_SHORT}],
        "messages": [{"role": "user", "content": [{"text": _user_text_v3(desc)}]}],
        "inferenceConfig": {"temperature": temperature, "topP": 0.9, "maxTokens": max_tokens},
    }
