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
    "Bạn là trợ lý ẩm thực chuyên trích xuất nguyên liệu TỪ ẢNH món ăn.\n"
    "Yêu cầu: trả về DUY NHẤT một JSON hợp lệ theo schema đã cho.\n"
    "Không trả lời giải thích, không thêm text ngoài JSON.\n"
    "Nếu thông tin không chắc chắn, để trống hoặc bỏ qua trường đó thay vì bịa.\n"
    "Ngôn ngữ ưu tiên: giữ tiếng Việt cho tên nguyên liệu.\n"
    "Tránh tên chung chung như 'thịt', 'rau' - hãy cụ thể như 'thịt bò thăn', 'rau mùi'.\n"
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
        + "quantity": chỉ chứa SỐ (ví dụ: "200", "1", "2")
        + "unit": chỉ chứa ĐƠN VỊ (ví dụ: "g", "ml", "củ", "nhánh", "quả", "tép")
    - Nếu không có đơn vị rõ ràng: unit = null

    Schema (JSON Schema):
    {schema_str}"""

    if include_example:
        example_str = json.dumps(FEW_SHOT_EXAMPLE, ensure_ascii=False)
        user_text += f"""

    Ví dụ (chỉ tham khảo, KHÔNG lẫn vào output):
    {example_str}"""

        user_text += """

    Lưu ý: Quantity LUÔN là chuỗi số ("1", "200", "0.5"), Unit là chuỗi đơn vị ("g", "ml", "củ") hoặc null.
    Trước khi trả: tự kiểm tra JSON hợp lệ theo schema. Nếu chưa hợp lệ, tự sửa rồi mới trả."""

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
    return {
        "inputText": f"{SYSTEM_INSTRUCTIONS}\n\n{user_text}",
        "textGenerationConfig": {
            "temperature": temperature,
            "topP": 0.9,
            "maxTokenCount": max_tokens,
            "stopSequences": [],
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

    user_text = f"""Bước 1: Nhìn vào ảnh, xác định tên món ăn cụ thể (ví dụ: "Phở bò", "Bún chả", "Cơm tấm", v.v.)
Bước 2: Sau khi xác định tên món, hãy tạo JSON nguyên liệu cho món đã nhận diện.

Mô tả bổ sung (nếu có):
\"\"\"{user_dish_description}\"\"\"

Yêu cầu định dạng:
- Trả về duy nhất một JSON.
- Tuân thủ schema (bên dưới) cả về key và kiểu.
- Tách quantity (chỉ SỐ) và unit (chỉ ĐƠN VỊ); nếu không có đơn vị: unit = null.

Schema:
{schema_str}

Ví dụ (tham khảo, KHÔNG lẫn vào output):
{example_str}

Lưu ý: Quantity là chuỗi số ("1","200","0.5"); Unit là chuỗi đơn vị ("g","ml","củ") hoặc null.
Tự kiểm tra JSON hợp lệ trước khi trả."""
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

    user_text = f"""Bước 1: Nhận diện tên món ăn trong ảnh (ví dụ: "Phở bò", "Bún chả", "Cơm tấm").
Bước 2: Xuất DUY NHẤT một JSON theo schema, tách quantity (số) và unit (đơn vị).

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
        '- unit: chỉ đơn vị ("g", "củ") hoặc null\n'
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
