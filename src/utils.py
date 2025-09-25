import logging, os
from dotenv import load_dotenv
import base64

load_dotenv()


def get_logger(name="app"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


REGION = os.getenv("AWS_REGION", "ap-southeast-1")
MODEL_ID = os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

print(f"Config: MODEL_ID={MODEL_ID}, REGION={REGION}, MAX_TOKENS={MAX_TOKENS}, TEMPERATURE={TEMPERATURE}")
def to_base64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")