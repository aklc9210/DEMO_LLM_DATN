from __future__ import annotations
import json
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .utils import get_logger, REGION


logger = get_logger("bedrock")


class BedrockError(RuntimeError): ...
class BedrockRateLimit(BedrockError): ...
class BedrockTimeout(BedrockError): ...
class BedrockInvalidResponse(BedrockError): ...


class BedrockClient:
    def __init__(self, region: str = REGION, timeout: int = 60):
        cfg = Config(
            region_name=region,
            retries={"max_attempts": 0},
            read_timeout=timeout,
            connect_timeout=10,
        )
        self.client = boto3.client("bedrock-runtime", config=cfg)

    def _classify(self, err: Exception) -> BedrockError:
        if isinstance(err, ClientError):
            code = err.response.get("Error", {}).get("Code", "")
            if code in {"ThrottlingException", "TooManyRequestsException"}:
                return BedrockRateLimit(str(err))
            if code in {"ModelTimeoutException", "GatewayTimeoutException"}:
                return BedrockTimeout(str(err))
        return BedrockError(str(err))

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.8, min=1, max=8),
        retry=retry_if_exception_type((BedrockRateLimit, BedrockTimeout, BotoCoreError, ClientError))
    )
    def invoke(self, model_id: str, body: dict, accept: str = "application/json",
                content_type: str = "application/json") -> dict:
        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body).encode("utf-8"),
                accept=accept,
                contentType=content_type,
            )
            payload = response.get("body")
            text = payload.read().decode("utf-8") if hasattr(payload, "read") else payload
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON from model: %s", text[:500])
                raise BedrockInvalidResponse(f"Model returned non-JSON: {e}")
        except (BotoCoreError, ClientError) as e:
            raise self._classify(e)