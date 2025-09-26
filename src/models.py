
IMAGE_MODELS = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude 3.5 Sonnet",
    "amazon.nova-pro-v1:0": "Amazon Nova Pro",
    "amazon.nova-lite-v1:0": "Amazon Nova Lite",
}

TEXT_MODELS = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude 3.5 Sonnet",
    "amazon.titan-text-lite-v1": "Titan Text G1 - Lite",
    "meta.llama3-8b-instruct-v1:0": "Llama 3 8B",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": "Claude 3.5 Haiku",
}

def get_default_max_tokens(model_id: str, default: int = 512) -> int:
    """Get default max tokens for a specific model."""
    if 'titan-text-lite' in model_id:
        return 1024  
    elif 'titan-text-express' in model_id:
        return 1024 
    elif 'claude-3-5-sonnet' in model_id:
        return 1024
    return default


PRICES_PER_1K_IN_OUT = {
    # Anthropic
    "anthropic.claude-3-5-sonnet-20240620-v1:0": (0.003, 0.015), 
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": (0.0008, 0.004),

    # Amazon Nova
    "amazon.nova-lite-v1:0": (0.00006, 0.00024),
    "amazon.nova-pro-v1:0":  (0.0008,  0.0032),

    # Amazon Titan Text
    "amazon.titan-text-lite-v1":    (0.00015, 0.0002),
    "amazon.titan-text-express-v1": (0.0002,  0.0006),

    # Meta
    "meta.llama3-8b-instruct-v1:0": (0.0003, 0.0006),
}

def get_model_cost_estimates(model_id: str) -> tuple[float, float]:
    return PRICES_PER_1K_IN_OUT.get(model_id, (0.0, 0.0))

def estimate_cost_simple(model_id: str, tokens_in: int, tokens_out: int) -> float:
    price_in_1k, price_out_1k = get_model_cost_estimates(model_id)
    return round((tokens_in/1000)*price_in_1k + (tokens_out/1000)*price_out_1k, 6)
