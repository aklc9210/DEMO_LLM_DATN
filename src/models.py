"""Model configurations and constants."""

IMAGE_MODELS = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude 3.5 Sonnet",
    "amazon.nova-pro-v1:0": "Amazon Nova Pro",
    "amazon.nova-lite-v1:0": "Amazon Nova Lite",
}

TEXT_MODELS = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude 3.5 Sonnet",
    "amazon.titan-text-lite-v1": "Titan Text G1 - Lite",
    "amazon.titan-text-express-v1": "Titan Text G1 - Express",
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

def get_model_cost_estimates(model_id: str) -> tuple[float, float]:
    """Get cost estimates (per 1k tokens) for input and output."""
    if 'claude' in model_id.lower():
        return 0.003, 0.015
    elif 'titan-text-lite' in model_id.lower():
        return 0.0003, 0.0004 
    elif 'titan-text-express' in model_id.lower():
        return 0.0005, 0.0008  
    return 0.001, 0.002