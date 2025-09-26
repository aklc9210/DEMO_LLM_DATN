from __future__ import annotations
import json
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .utils import get_logger, REGION
from botocore.response import StreamingBody


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

    def _headers_lower(self, resp) -> dict:
        try:
            return {k.lower(): v for k, v in resp["ResponseMetadata"]["HTTPHeaders"].items()}
        except Exception:
            return {}

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.8, min=1, max=8),
        retry=retry_if_exception_type((BedrockRateLimit, BedrockTimeout, BotoCoreError, ClientError))
    )
    def invoke(self, model_id: str, body: dict,
            accept: str = "application/json",
            content_type: str = "application/json") -> tuple[dict, dict]:
        """
        Trả về (json_result, headers_lowercased).
        Headers chứa token thật: x-amzn-bedrock-input-token-count, x-amzn-bedrock-output-token-count
        """
        try:
            logger.info(f"Invoking model: {model_id}")
            resp = self.client.invoke_model(
                modelId=model_id,
                accept=accept,
                contentType=content_type,
                body=json.dumps(body)
            )
            headers = self._headers_lower(resp)
            # body có thể là StreamingBody
            raw = resp.get("body")
            if isinstance(raw, StreamingBody):
                text = raw.read()
                text = text.decode("utf-8", errors="ignore")
            else:
                text = raw if isinstance(raw, str) else ""
            if not text or text.strip() == "":
                raise BedrockInvalidResponse("Bedrock returned empty response")
            try:
                return json.loads(text), headers
            except json.JSONDecodeError as e:
                raise BedrockInvalidResponse(f"Model returned non-JSON: {e}")
        except (BotoCoreError, ClientError) as e:
            raise self._classify(e)


    def count_tokens(self, model_id: str, request_body: dict, content_type: str = "application/json") -> int:
        """
        Dùng Bedrock CountTokens để đếm input tokens CHUẨN trước khi invoke.
        Không tính phí. Kết quả bằng đúng số sẽ bị tính tiền khi invoke cùng nội dung.
        """
        try:
            resp = self.client.count_tokens(
                modelId=model_id,
                contentType=content_type,
                body=json.dumps(request_body)
            )
            usage = resp.get("body") 

            if isinstance(usage, (bytes, str)):
                usage = json.loads(usage if isinstance(usage, str) else usage.decode("utf-8"))
            return int(usage.get("totalTokens") or usage.get("inputTokens") or 0)
        except Exception as e:
            logger.warning(f"CountTokens failed: {e}")
            return 0